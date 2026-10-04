"""Estimation de l'exposant de la loi de Zipf (loi de Pareto) à partir d'une liste de
tailles.

Convention : l'exposant ζ est celui de la fonction de survie, P(X > x) ∝ x^(-ζ). La loi
de Zipf proprement dite correspond à ζ = 1 : la ville de rang r a une taille
proportionnelle à 1/r. (L'exposant α de la *densité* de Clauset, Shalizi et Newman vaut
ζ + 1.)

Trois méthodes :

1. ``ajuste_mco`` : régression de log(rang) sur log(taille) (la méthode « classique » de
   la littérature). Avec ``decalage=0.5`` c'est la correction de Gabaix et Ibragimov
   (2011), qui réduit fortement le biais en petit échantillon. Les écarts-types de la
   régression sont faux, car les points ne sont pas indépendants (les rangs sont triés)
   : on fournit aussi l'écart-type corrigé de Gabaix et Ibragimov, ζ·√(2/n).
2. ``exposant_pareto`` : maximum de vraisemblance (estimateur de Hill) au-dessus d'un
   seuil ``xmin``. ``choisit_xmin`` choisit le seuil en minimisant la distance de
   Kolmogorov-Smirnov entre les données et la loi ajustée (Clauset, Shalizi, Newman,
   2009).
3. ``compare_lognormale`` : test du rapport de vraisemblance de Vuong entre la loi de
   Pareto et la loi log-normale, toutes deux tronquées en ``xmin``.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


def _decroissant(tailles):
    x = np.sort(np.asarray(tailles, dtype=float))[::-1]
    if len(x) == 0 or x[-1] <= 0:
        raise ValueError("les tailles doivent être strictement positives")
    return x


# --- 1. régression log-rang / log-taille ----------------------------------------------


@dataclass(frozen=True)
class AjustementMCO:
    exposant: float
    erreur_mco: float  # écart-type « naïf » de la régression (non valide ici)
    erreur_gabaix: float  # écart-type corrigé : exposant * sqrt(2 / n)
    n: int
    r2: float
    durbin_watson: float  # proche de 0 : résidus très autocorrélés


def ajuste_mco(tailles, n=None, decalage=0.0):
    """Régresse ``log(rang - decalage)`` sur ``log(taille)`` pour les ``n`` plus grandes
    tailles (toutes par défaut). ``decalage=0.5`` : correction de Gabaix et Ibragimov.
    """
    x = _decroissant(tailles)
    if n is not None:
        x = x[:n]
    n = len(x)
    log_taille = np.log(x)
    y = np.log(np.arange(1, n + 1) - decalage)

    xc = log_taille - log_taille.mean()
    pente = (xc * (y - y.mean())).sum() / (xc**2).sum()
    ordonnee = y.mean() - pente * log_taille.mean()
    residus = y - (ordonnee + pente * log_taille)

    sce = (residus**2).sum()
    variance = sce / (n - 2)
    exposant = -pente
    return AjustementMCO(
        exposant=exposant,
        erreur_mco=float(np.sqrt(variance / (xc**2).sum())),
        erreur_gabaix=float(exposant * np.sqrt(2 / n)),
        n=n,
        r2=float(1 - sce / ((y - y.mean()) ** 2).sum()),
        durbin_watson=float((np.diff(residus) ** 2).sum() / sce),
    )


# --- 2. maximum de vraisemblance ------------------------------------------------------


@dataclass(frozen=True)
class AjustementPareto:
    exposant: float
    erreur: float  # exposant / sqrt(n)
    xmin: float
    n: int  # nombre de tailles >= xmin
    ks: float  # distance de Kolmogorov-Smirnov


def exposant_pareto(tailles, xmin):
    """Maximum de vraisemblance de ζ sur les tailles >= xmin : n / Σ ln(x_i / xmin)."""
    x = np.asarray(tailles, dtype=float)
    queue = np.sort(x[x >= xmin])
    n = len(queue)
    if n < 2:
        raise ValueError("moins de deux tailles au-dessus de xmin")
    exposant = n / np.log(queue / xmin).sum()
    return AjustementPareto(
        exposant=float(exposant),
        erreur=float(exposant / np.sqrt(n)),
        xmin=float(xmin),
        n=n,
        ks=_distance_ks(queue, xmin, exposant),
    )


def _distance_ks(queue_croissante, xmin, exposant):
    n = len(queue_croissante)
    modele = 1 - (xmin / queue_croissante) ** exposant
    empirique_haut = np.arange(1, n + 1) / n
    empirique_bas = np.arange(0, n) / n
    return float(
        max(np.abs(empirique_haut - modele).max(), np.abs(empirique_bas - modele).max())
    )


def choisit_xmin(tailles, n_min=30, n_candidats=200):
    """Seuil ``xmin`` qui minimise la distance de Kolmogorov-Smirnov.

    Les candidats sont les tailles des villes de rang ``n_min`` à N, espacés
    géométriquement (plus serrés vers le haut du classement).
    """
    x = _decroissant(tailles)
    rangs = np.unique(np.geomspace(n_min, len(x), n_candidats).astype(int))
    meilleur = None
    for k in rangs:
        xmin = x[k - 1]
        try:
            ajustement = exposant_pareto(x, xmin)
        except ValueError:
            continue
        if meilleur is None or ajustement.ks < meilleur.ks:
            meilleur = ajustement
    if meilleur is None:
        raise ValueError("pas assez de données pour choisir xmin")
    return meilleur.xmin


def ajuste_pareto(tailles, xmin=None, n_min=30):
    """Ajustement de Pareto ; ``xmin`` est choisi par ``choisit_xmin`` s'il est omis."""
    if xmin is None:
        xmin = choisit_xmin(tailles, n_min=n_min)
    return exposant_pareto(tailles, xmin)


# --- 3. comparaison avec la loi log-normale -------------------------------------------


def loglik_pareto(queue, xmin, exposant):
    """Log-vraisemblance de chaque taille sous la loi de Pareto de seuil xmin."""
    return np.log(exposant) + exposant * np.log(xmin) - (exposant + 1) * np.log(queue)


def ajuste_lognormale_tronquee(queue, xmin):
    """Maximum de vraisemblance (mu, sigma) d'une loi log-normale tronquée à xmin."""
    logs = np.log(queue)
    log_xmin = np.log(xmin)

    def moins_loglik(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        z = (logs - mu) / sigma
        survie = stats.norm.logsf((log_xmin - mu) / sigma)
        return -(
            -logs - log_sigma - 0.5 * z**2 - 0.5 * np.log(2 * np.pi) - survie
        ).sum()

    depart = [logs.mean(), np.log(max(logs.std(), 1e-3))]
    res = optimize.minimize(moins_loglik, depart, method="Nelder-Mead")
    return float(res.x[0]), float(np.exp(res.x[1]))


def loglik_lognormale_tronquee(queue, xmin, mu, sigma):
    logs = np.log(queue)
    z = (logs - mu) / sigma
    survie = stats.norm.logsf((np.log(xmin) - mu) / sigma)
    return -logs - np.log(sigma) - 0.5 * z**2 - 0.5 * np.log(2 * np.pi) - survie


@dataclass(frozen=True)
class ComparaisonLoiPuissanceLognormale:
    xmin: float
    n: int
    exposant: float
    mu: float
    sigma: float
    rapport: float  # somme des différences de log-vraisemblance (> 0 : Pareto préférée)
    statistique: float  # rapport normalisé de Vuong
    p_value: float  # bilatérale : un p élevé signifie « indécidable »
    delta_aic: float  # AIC(Pareto) - AIC(log-normale) (< 0 : Pareto préférée)


def compare_lognormale(tailles, xmin):
    """Test de Vuong : Pareto contre log-normale, toutes deux tronquées à xmin."""
    x = np.asarray(tailles, dtype=float)
    queue = x[x >= xmin]
    pareto = exposant_pareto(queue, xmin)
    mu, sigma = ajuste_lognormale_tronquee(queue, xmin)

    l_pareto = loglik_pareto(queue, xmin, pareto.exposant)
    l_lognormale = loglik_lognormale_tronquee(queue, xmin, mu, sigma)
    difference = l_pareto - l_lognormale
    n = len(queue)
    rapport = float(difference.sum())
    ecart = float(difference.std())
    statistique = rapport / (ecart * np.sqrt(n)) if ecart > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(statistique)))
    aic_pareto = 2 * 1 - 2 * l_pareto.sum()  # 1 paramètre
    aic_lognormale = 2 * 2 - 2 * l_lognormale.sum()  # 2 paramètres
    return ComparaisonLoiPuissanceLognormale(
        xmin=float(xmin),
        n=n,
        exposant=pareto.exposant,
        mu=mu,
        sigma=sigma,
        rapport=rapport,
        statistique=float(statistique),
        p_value=p_value,
        delta_aic=float(aic_pareto - aic_lognormale),
    )
