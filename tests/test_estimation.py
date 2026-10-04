import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from zipf_villes.estimation import (
    ajuste_lognormale_tronquee,
    ajuste_mco,
    ajuste_pareto,
    choisit_xmin,
    compare_lognormale,
    exposant_pareto,
)


def pareto(rng, zeta, n, xmin=1.0):
    """Échantillon de loi de Pareto : P(X > x) = (xmin / x)^zeta pour x >= xmin."""
    return xmin * (rng.pareto(zeta, n) + 1)


# --- régression log-rang / log-taille -------------------------------------------------


def test_mco_matches_statsmodels():
    rng = np.random.default_rng(0)
    tailles = rng.lognormal(10, 1.5, 500)
    ajustement = ajuste_mco(tailles, n=200)

    haut = np.sort(tailles)[::-1][:200]
    modele = sm.OLS(np.log(np.arange(1, 201)), sm.add_constant(np.log(haut))).fit()
    assert ajustement.exposant == pytest.approx(-modele.params[1])
    assert ajustement.erreur_mco == pytest.approx(modele.bse[1])
    assert ajustement.r2 == pytest.approx(modele.rsquared)
    assert ajustement.durbin_watson == pytest.approx(durbin_watson(modele.resid))
    assert ajustement.n == 200


def test_mco_residuals_are_strongly_autocorrelated():
    """C'est pourquoi les écarts-types de la régression ne valent rien."""
    tailles = pareto(np.random.default_rng(1), 1.0, 300)
    assert ajuste_mco(tailles).durbin_watson < 1.0  # 2 = pas d'autocorrélation


def test_gabaix_ibragimov_reduces_small_sample_bias():
    rng = np.random.default_rng(2)
    classique, corrige = [], []
    for _ in range(300):
        tailles = pareto(rng, 1.0, 100)
        classique.append(ajuste_mco(tailles).exposant)
        corrige.append(ajuste_mco(tailles, decalage=0.5).exposant)
    assert abs(np.mean(corrige) - 1) < 0.03
    assert abs(np.mean(corrige) - 1) < abs(np.mean(classique) - 1)


def test_gabaix_standard_error_formula():
    ajustement = ajuste_mco(pareto(np.random.default_rng(3), 1.0, 400))
    assert ajustement.erreur_gabaix == pytest.approx(
        ajustement.exposant * np.sqrt(2 / 400)
    )


def test_mco_does_not_modify_its_input():
    tailles = np.array([5.0, 100.0, 20.0, 1.0, 50.0])
    copie = tailles.copy()
    ajuste_mco(tailles)
    np.testing.assert_array_equal(tailles, copie)


@pytest.mark.parametrize("tailles", [[1.0, 0.0, 3.0], [1.0, -2.0], []])
def test_non_positive_sizes_are_rejected(tailles):
    with pytest.raises(ValueError):
        ajuste_mco(tailles)


# --- maximum de vraisemblance ---------------------------------------------------------


def test_pareto_mle_recovers_exponent():
    tailles = pareto(np.random.default_rng(0), 1.5, 5000, xmin=100)
    ajustement = exposant_pareto(tailles, xmin=100)
    assert ajustement.exposant == pytest.approx(1.5, abs=0.07)
    assert ajustement.erreur == pytest.approx(ajustement.exposant / np.sqrt(5000))
    assert ajustement.n == 5000
    assert ajustement.ks < 0.03


def test_pareto_mle_formula():
    # n = 3 ; ln(2) + ln(4) + ln(8) = 6 ln 2 ; exposant = 3 / (6 ln 2)
    assert exposant_pareto([2.0, 4.0, 8.0], xmin=1.0).exposant == pytest.approx(
        3 / (6 * np.log(2))
    )


def test_pareto_mle_ignores_values_below_xmin():
    tailles = pareto(np.random.default_rng(0), 1.5, 2000, xmin=100)
    avec_bruit = np.concatenate([tailles, [1.0, 5.0, 50.0]])
    assert (
        exposant_pareto(avec_bruit, 100).exposant
        == exposant_pareto(tailles, 100).exposant
    )


def test_xmin_selection_finds_the_start_of_the_tail():
    rng = np.random.default_rng(0)
    corps = rng.lognormal(3, 1, 3000)
    corps = corps[corps < 100]
    queue = pareto(rng, 1.5, 1000, xmin=100)
    melange = np.concatenate([corps, queue])

    xmin = choisit_xmin(melange)
    assert 60 < xmin < 250
    assert ajuste_pareto(melange).exposant == pytest.approx(1.5, abs=0.15)


def test_pareto_needs_enough_points():
    with pytest.raises(ValueError):
        exposant_pareto([1.0, 2.0, 3.0], xmin=10.0)


# --- loi log-normale ------------------------------------------------------------------


def test_truncated_lognormal_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    tailles = rng.lognormal(5, 1.2, 6000)
    xmin = np.quantile(tailles, 0.5)
    mu, sigma = ajuste_lognormale_tronquee(tailles[tailles >= xmin], xmin)
    assert mu == pytest.approx(5, abs=0.3)
    assert sigma == pytest.approx(1.2, abs=0.15)


def test_vuong_prefers_lognormal_on_lognormal_data():
    rng = np.random.default_rng(0)
    tailles = rng.lognormal(5, 1.2, 6000)
    comparaison = compare_lognormale(tailles, np.quantile(tailles, 0.5))
    assert comparaison.rapport < 0
    assert comparaison.p_value < 1e-6
    assert comparaison.delta_aic > 0


def test_vuong_cannot_reject_pareto_on_pareto_data():
    """Une log-normale très étalée imite une loi de puissance : le test est indécis."""
    tailles = pareto(np.random.default_rng(0), 1.5, 5000, xmin=100)
    comparaison = compare_lognormale(tailles, 100)
    assert comparaison.p_value > 0.05
    assert comparaison.exposant == pytest.approx(1.5, abs=0.07)


# --- contrôle croisé avec le package de référence --------------------------------


def test_matches_the_powerlaw_package():
    powerlaw = pytest.importorskip("powerlaw")
    rng = np.random.default_rng(0)

    tailles = pareto(rng, 1.5, 5000, xmin=100)
    reference = powerlaw.Fit(tailles, xmin=100, discrete=False, verbose=False)
    assert exposant_pareto(tailles, 100).exposant == pytest.approx(
        reference.power_law.alpha - 1  # alpha de powerlaw : exposant de la densité
    )

    tailles = rng.lognormal(5, 1.2, 6000)
    xmin = np.quantile(tailles, 0.5)
    reference = powerlaw.Fit(tailles, xmin=xmin, discrete=False, verbose=False)
    rapport, p_value = reference.distribution_compare("power_law", "lognormal")
    comparaison = compare_lognormale(tailles, xmin)
    assert comparaison.rapport == pytest.approx(rapport, rel=1e-4)
    assert comparaison.p_value == pytest.approx(p_value, rel=1e-3)
