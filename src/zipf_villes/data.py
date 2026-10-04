"""Chargement et préparation des populations de villes (France et États-Unis).

Les fonctions de préparation (``populations_*``) sont pures : elles prennent des
DataFrames bruts et en renvoient de nouveaux, sans toucher à leurs arguments ni au
disque. Seules les fonctions ``lit_*`` et ``charge_populations`` lisent ou téléchargent
des fichiers.

Sources :

- communes : Insee, population en 2020
  https://www.insee.fr/fr/statistiques/6683035?sommaire=6683037
- aires d'attraction des villes : Insee, zonage 2020
  https://www.insee.fr/fr/information/4803954
- aires métropolitaines américaines (MSA) : US Census Bureau, estimations 2021
  https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-metro-and-micro-statistical-areas.html
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL_COMMUNES = "https://www.insee.fr/fr/statistiques/fichier/6683035/ensemble.zip"
URL_AAV = (
    "https://www.insee.fr/fr/statistiques/fichier/4803954/AAV2020_au_01-01-2022.zip"
)
FICHIER_AAV = "AAV2020_au_01-01-2022.xlsx"
URL_MSA = (
    "https://www2.census.gov/programs-surveys/popest/tables/2020-2021/metro/totals/"
)
FICHIER_MSA = "cbsa-met-est2021-pop.xlsx"

# Communes à arrondissements : code commune -> codes des arrondissements
ARRONDISSEMENTS = {
    "75056": [f"751{n:02}" for n in range(1, 21)],  # Paris
    "69123": [f"6938{n}" for n in range(1, 10)],  # Lyon
    "13055": [f"132{n:02}" for n in range(1, 17)],  # Marseille
}


# --- préparation (fonctions pures) ----------------------------------------------------


def ajoute_rang(df, colonne):
    """Trie par ``colonne`` décroissante et ajoute ``rang`` (1 pour la plus grande),
    ``log_rang`` et ``log_pop`` (logarithme népérien de ``colonne``)."""
    df = df.sort_values(colonne, ascending=False).reset_index(drop=True)
    return df.assign(
        rang=df.index + 1,
        log_rang=np.log(df.index + 1),
        log_pop=np.log(df[colonne]),
    )


def agrege_arrondissements(communes):
    """Rattache les arrondissements de Paris, Lyon et Marseille à leur commune."""
    vers_commune = {
        arrondissement: commune
        for commune, arrondissements in ARRONDISSEMENTS.items()
        for arrondissement in arrondissements
    }
    return communes.assign(
        CODGEO=communes["CODGEO"].replace(vers_commune),
        COM=communes["COM"].replace("(Paris|Lyon|Marseille) .+", r"\1", regex=True),
    )


def populations_communes(brut):
    """Population des communes à partir de ``donnees_communes.csv`` de l'Insee.

    La population des arrondissements de Paris, Lyon et Marseille est agrégée pour
    obtenir celle de la commune.
    """
    codgeo = brut["CODDEP"].astype(str) + brut["CODCOM"].astype(str).str.pad(
        3, fillchar="0"
    )
    communes = brut.assign(CODGEO=codgeo)[["COM", "CODGEO", "PTOT"]]
    somme = (
        agrege_arrondissements(communes)
        .groupby(["COM", "CODGEO"])["PTOT"]
        .sum()
        .loc[lambda s: s > 0]
        .reset_index()
    )
    return ajoute_rang(somme, "PTOT")


def populations_aav(composition, com_pop):
    """Population des aires d'attraction des villes.

    ``composition`` : feuille « Composition_communale » du fichier de l'Insee ;
    ``com_pop`` : résultat de ``populations_communes``. Les communes d'outre-mer (code
    commençant par 97) sont exclues et les communes situées hors de toute aire
    d'attraction (code d'aire « 000 ») ne forment pas une ville : elles sont retirées
    *avant* le classement, pour que les rangs se suivent.
    """
    # zfill : protège d'une lecture des codes comme des entiers (« 000 » devenu 0)
    composition = composition.assign(
        CODGEO=composition["CODGEO"].astype(str).str.zfill(5),
        AAV2020=composition["AAV2020"].astype(str).str.zfill(3),
    )
    somme = (
        composition.loc[~composition["CODGEO"].str.startswith("97")]
        .merge(com_pop[["CODGEO", "PTOT"]], on="CODGEO", how="left")
        .groupby(["AAV2020", "LIBAAV2020"])["PTOT"]
        .sum()
        .loc[lambda s: s > 0]
        .reset_index()
        .loc[lambda s: s["AAV2020"] != "000"]
    )
    return ajoute_rang(somme, "PTOT")


def populations_msa(brut):
    """Population des aires métropolitaines américaines (colonne ``pop``, année 2021).

    Le fichier du Census liste aussi les *divisions métropolitaines* (noms contenant
    « Metro Division »), qui sont des sous-ensembles d'une aire métropolitaine : les
    garder compterait deux fois une même population. Elles sont retirées. Les points
    qui précèdent les noms dans le fichier (niveau d'indentation) sont enlevés.
    """
    msa = (
        brut.drop(columns="Unnamed: 1")
        .rename(columns={"Unnamed: 0": "msa"})
        .drop([0, 1])
        .dropna()
        .assign(msa=lambda s: s["msa"].str.lstrip("."))
        .loc[lambda s: ~s["msa"].str.contains("Metro Division")]
        .set_index("msa")[2021]
        .rename("pop")
        .reset_index()
    )
    return ajoute_rang(msa, "pop")


# --- lecture et téléchargement --------------------------------------------------------


def _telecharge(url):
    with urllib.request.urlopen(url) as reponse:
        return reponse.read()


def lit_communes(data_path):
    dossier = Path(data_path) / "communes"
    if not dossier.exists():
        zipfile.ZipFile(io.BytesIO(_telecharge(URL_COMMUNES))).extractall(dossier)
    return pd.read_csv(
        dossier / "donnees_communes.csv",
        sep=";",
        dtype={"CODDEP": str, "CODCOM": str},  # garde les zéros de « 01 » et « 053 »
    )


def lit_composition_aav(data_path):
    dossier = Path(data_path) / "aav"
    if not dossier.exists():
        zipfile.ZipFile(io.BytesIO(_telecharge(URL_AAV))).extractall(dossier)
    return pd.read_excel(
        dossier / FICHIER_AAV,
        skiprows=5,
        sheet_name="Composition_communale",
        # sans cela pandas lit « 000 » comme l'entier 0 et « 01053 » comme 1053
        dtype={"CODGEO": str, "AAV2020": str},
    )


def lit_msa(data_path):
    dossier = Path(data_path) / "usa"
    if not dossier.exists():
        dossier.mkdir(parents=True)
        (dossier / FICHIER_MSA).write_bytes(_telecharge(URL_MSA + FICHIER_MSA))
    return pd.read_excel(dossier / FICHIER_MSA, skiprows=3)


def charge_populations(data_path="data"):
    """Les trois jeux de données, prêts à analyser (téléchargés au premier appel).

    Renvoie ``(com_pop, aav, msa)``.
    """
    com_pop = populations_communes(lit_communes(data_path))
    aav = populations_aav(lit_composition_aav(data_path), com_pop)
    msa = populations_msa(lit_msa(data_path))
    return com_pop, aav, msa
