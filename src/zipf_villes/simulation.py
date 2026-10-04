"""Simulation d'un processus de croissance de villes qui produit une loi de Zipf.

Chaque ville croît à chaque période d'un taux aléatoire ε ~ N(mu, sigma), indépendant de
sa taille (loi de Gibrat) ; une ville ne peut pas descendre sous une taille minimale
``seuil`` (barrière inférieure). Gabaix (1999) a montré qu'alors la distribution des
tailles converge vers une loi de Pareto d'exposant ζ, solution de E[(1 + ε)^ζ] = 1.
Quand le taux de croissance moyen est nul (mu = 0) on a E[1 + ε] = 1 : ζ = 1, c'est la
loi de Zipf.

Le temps de convergence est long : il croît comme 1/sigma². Avec sigma = 0.1 il faut
plusieurs centaines de périodes pour que la queue de la distribution se stabilise.
"""

import numpy as np
from scipy import integrate, optimize, stats


def simule_villes(
    n_villes=2000, periodes=100, sigma=0.1, mu=0.0, seuil=10_000, initial=None, rng=None
):
    """Tailles de ``n_villes`` villes sur ``periodes`` périodes, tableau (periodes,
    n_villes).

    ``initial`` : tailles de départ (toutes égales à ``seuil`` par défaut) ;
    ``rng`` : graine entière ou ``numpy.random.Generator``.
    """
    rng = np.random.default_rng(rng)
    tailles = np.zeros((periodes, n_villes))
    if initial is None:
        tailles[0] = seuil
    else:
        tailles[0] = np.maximum(initial, seuil)
    for t in range(1, periodes):
        croissance = 1 + rng.normal(mu, sigma, n_villes)
        tailles[t] = np.maximum(tailles[t - 1] * croissance, seuil)
    return tailles


def exposant_theorique(mu=0.0, sigma=0.1):
    """Exposant ζ de la loi limite : racine de E[(1 + ε)^ζ] = 1 avec ε ~ N(mu, sigma).

    Les valeurs ε ≤ -1 (ville qui disparaîtrait) tombent sur la barrière et ne comptent
    pas dans l'espérance. Renvoie ``nan`` si la taille logarithmique moyenne ne décroît
    pas (pas de loi limite).
    """

    def esperance(zeta):
        valeur, _ = integrate.quad(
            lambda e: (1 + e) ** zeta * stats.norm.pdf(e, mu, sigma),
            -1,
            np.inf,
            limit=200,
        )
        return valeur - 1

    derive, _ = integrate.quad(
        lambda e: np.log1p(e) * stats.norm.pdf(e, mu, sigma), -1, np.inf, limit=200
    )
    if derive >= 0:
        return float("nan")
    return float(optimize.brentq(esperance, 1e-3, 50))
