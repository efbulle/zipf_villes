import numpy as np
import pytest

from zipf_villes.estimation import ajuste_pareto
from zipf_villes.simulation import exposant_theorique, simule_villes


def test_shape_and_barrier():
    tailles = simule_villes(n_villes=50, periodes=30, seuil=10_000, rng=1)
    assert tailles.shape == (30, 50)
    assert tailles.min() >= 10_000
    np.testing.assert_array_equal(tailles[0], 10_000)  # tout le monde part du seuil


def test_reproducible_with_a_seed():
    a = simule_villes(n_villes=20, periodes=10, rng=42)
    b = simule_villes(n_villes=20, periodes=10, rng=42)
    c = simule_villes(n_villes=20, periodes=10, rng=43)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_initial_sizes_are_used_and_not_modified():
    initial = np.array([5_000.0, 20_000.0, 80_000.0])
    copie = initial.copy()
    tailles = simule_villes(
        n_villes=3, periodes=2, initial=initial, seuil=10_000, rng=0
    )
    np.testing.assert_array_equal(initial, copie)
    assert tailles[0, 0] == 10_000  # remonté au seuil
    assert tailles[0, 1] == 20_000


def test_theoretical_exponent():
    # mu = 0 : E[1 + eps] = 1 donc zeta = 1 (à la correction près des eps <= -1)
    assert exposant_theorique(0.0, 0.03) == pytest.approx(1.0, abs=1e-3)
    assert exposant_theorique(0.0, 0.1) == pytest.approx(1.0, abs=1e-3)
    assert exposant_theorique(0.0, 0.5) == pytest.approx(0.969, abs=0.005)
    # croissance moyenne négative : exposant plus grand (villes plus égales)
    assert exposant_theorique(-0.01, 0.1) > 1.1


def test_no_limit_law_when_cities_drift_upward():
    assert np.isnan(exposant_theorique(0.02, 0.1))


def test_long_simulation_converges_to_zipf():
    tailles = simule_villes(n_villes=2000, periodes=2000, sigma=0.1, rng=1235)
    assert ajuste_pareto(tailles[-1]).exposant == pytest.approx(1.0, abs=0.1)


def test_short_simulation_has_not_converged():
    """Le temps de convergence est long : à 100 périodes la queue est trop fine."""
    tailles = simule_villes(n_villes=2000, periodes=100, sigma=0.1, rng=1235)
    assert ajuste_pareto(tailles[-1]).exposant > 1.4
