"""Analyses statistiques de la loi de Zipf sur les populations de villes."""

from .data import charge_populations
from .estimation import (
    ajuste_mco,
    ajuste_pareto,
    choisit_xmin,
    compare_lognormale,
    exposant_pareto,
)
from .simulation import exposant_theorique, simule_villes

__all__ = [
    "ajuste_mco",
    "ajuste_pareto",
    "charge_populations",
    "choisit_xmin",
    "compare_lognormale",
    "exposant_pareto",
    "exposant_theorique",
    "simule_villes",
]
