import io
import zipfile

import numpy as np
import pandas as pd

from zipf_villes import data
from zipf_villes.data import (
    agrege_arrondissements,
    ajoute_rang,
    charge_populations,
    populations_aav,
    populations_communes,
    populations_msa,
)

COLONNES = ["rang", "log_rang", "log_pop"]


def communes_brutes():
    lignes = [
        ("Paris 1er Arrondissement", "75", "101", 16_000),
        ("Paris 2e Arrondissement", "75", "102", 20_000),
        ("Lyon 1er Arrondissement", "69", "381", 30_000),
        ("Lyon 2e Arrondissement", "69", "382", 30_000),
        ("Marseille 1er Arrondissement", "13", "201", 40_000),
        ("Bourg", "01", "53", 40_000),
        ("Village", "01", "1", 120),
        ("Fantôme", "01", "2", 0),
        # une commune corse : comme dans le vrai fichier, la colonne CODDEP contient des
        # lettres, donc pandas la lit comme du texte et garde les zéros de « 01 »
        ("Ajaccio", "2A", "4", 70_000),
    ]
    return pd.DataFrame(lignes, columns=["COM", "CODDEP", "CODCOM", "PTOT"])


def test_ajoute_rang():
    df = pd.DataFrame({"nom": ["a", "b", "c"], "pop": [10, 1000, 100]})
    copie = df.copy()
    resultat = ajoute_rang(df, "pop")
    assert list(resultat["nom"]) == ["b", "c", "a"]
    assert list(resultat["rang"]) == [1, 2, 3]
    np.testing.assert_allclose(resultat["log_rang"], np.log([1, 2, 3]))
    np.testing.assert_allclose(resultat["log_pop"], np.log([1000, 100, 10]))
    pd.testing.assert_frame_equal(df, copie)  # pas d'effet de bord


def test_agrege_arrondissements_is_pure():
    brut = pd.DataFrame(
        {
            "COM": ["Paris 1er Arrondissement", "Bourg"],
            "CODGEO": ["75101", "01053"],
            "PTOT": [1, 2],
        }
    )
    copie = brut.copy()
    resultat = agrege_arrondissements(brut)
    assert list(resultat["CODGEO"]) == ["75056", "01053"]
    assert list(resultat["COM"]) == ["Paris", "Bourg"]
    pd.testing.assert_frame_equal(brut, copie)


def test_populations_communes():
    brut = communes_brutes()
    copie = brut.copy()
    pop = populations_communes(brut)
    pd.testing.assert_frame_equal(brut, copie)

    assert list(pop.columns) == ["COM", "CODGEO", "PTOT", *COLONNES]
    assert "Fantôme" not in set(pop["COM"])  # population nulle écartée
    par_commune = dict(zip(pop["COM"], pop["PTOT"], strict=True))
    assert par_commune == {
        "Ajaccio": 70_000,
        "Paris": 36_000,
        "Lyon": 60_000,
        "Marseille": 40_000,
        "Bourg": 40_000,
        "Village": 120,
    }
    assert list(pop["rang"]) == [1, 2, 3, 4, 5, 6]
    assert pop.loc[pop["COM"] == "Ajaccio", "CODGEO"].item() == "2A004"
    assert pop.loc[pop["COM"] == "Paris", "CODGEO"].item() == "75056"
    assert pop.loc[pop["COM"] == "Village", "CODGEO"].item() == "01001"


def composition_aav():
    lignes = [
        ("75056", "001", "Paris"),
        ("01053", "002", "Bourg-en-Bresse"),
        ("01001", "002", "Bourg-en-Bresse"),
        ("2A004", "005", "Ajaccio"),
        ("69123", "003", "Lyon"),
        ("13055", "000", "Hors attraction des villes"),  # ne forme pas une ville
        ("97101", "004", "Outre-mer"),
    ]
    return pd.DataFrame(lignes, columns=["CODGEO", "AAV2020", "LIBAAV2020"])


def test_populations_aav_excludes_outside_and_overseas_before_ranking():
    com_pop = populations_communes(communes_brutes())
    composition = composition_aav()
    aav = populations_aav(composition, com_pop)

    assert list(aav["LIBAAV2020"]) == ["Ajaccio", "Lyon", "Bourg-en-Bresse", "Paris"]
    assert list(aav["PTOT"]) == [70_000, 60_000, 40_120, 36_000]
    # « Marseille » (la plus grande population de la zone 000) est retirée AVANT le
    # classement : les rangs se suivent, sans trou
    assert list(aav["rang"]) == [1, 2, 3, 4]
    assert "000" not in set(aav["AAV2020"])


def census_brut():
    """Imite la feuille lue par pandas avec ``skiprows=3``."""
    return pd.DataFrame(
        {
            "Unnamed: 0": [
                "Metropolitan Statistical Area",
                None,
                ".New York-Newark-Jersey City, NY-NJ-PA Metro Area",
                "..New York-Jersey City-White Plains, NY-NJ Metro Division",
                ".Los Angeles-Long Beach-Anaheim, CA Metro Area",
                "..Los Angeles-Long Beach-Glendale, CA Metro Division",
                "..Guayama, PR Metro Area",
                "Note : source Census Bureau",
            ],
            "Unnamed: 1": [
                None,
                None,
                19_000_000,
                12_000_000,
                13_000_000,
                9_800_000,
                69_000,
                None,
            ],
            2020: [
                None,
                None,
                19_500_000,
                12_100_000,
                13_100_000,
                9_850_000,
                68_500,
                None,
            ],
            2021: [
                None,
                None,
                19_768_458,
                12_076_970,
                12_997_353,
                9_829_544,
                68_178,
                None,
            ],
        }
    )


def test_populations_msa_drops_metro_divisions():
    brut = census_brut()
    copie = brut.copy()
    msa = populations_msa(brut)
    pd.testing.assert_frame_equal(brut, copie)

    assert list(msa.columns) == ["msa", "pop", *COLONNES]
    assert list(msa["msa"]) == [
        "New York-Newark-Jersey City, NY-NJ-PA Metro Area",
        "Los Angeles-Long Beach-Anaheim, CA Metro Area",
        "Guayama, PR Metro Area",  # garde les aires de Porto Rico (deux points aussi)
    ]
    assert list(msa["pop"]) == [19_768_458, 12_997_353, 68_178]
    assert list(msa["rang"]) == [1, 2, 3]


# --- lecture de fichiers (faux fichiers, sans réseau) ---------------------------------


def ecrit_faux_fichiers(racine):
    (racine / "communes").mkdir()
    communes_brutes().to_csv(
        racine / "communes" / "donnees_communes.csv", sep=";", index=False
    )

    (racine / "aav").mkdir()
    with pd.ExcelWriter(racine / "aav" / data.FICHIER_AAV) as xl:
        composition_aav().to_excel(
            xl, sheet_name="Composition_communale", startrow=5, index=False
        )

    (racine / "usa").mkdir()
    brut = census_brut()
    with pd.ExcelWriter(racine / "usa" / data.FICHIER_MSA) as xl:
        brut.rename(columns={"Unnamed: 0": None, "Unnamed: 1": None}).to_excel(
            xl, startrow=3, index=False
        )


def test_charge_populations_from_local_files(tmp_path):
    ecrit_faux_fichiers(tmp_path)
    com_pop, aav, msa = charge_populations(tmp_path)
    assert list(com_pop["COM"])[:2] == ["Ajaccio", "Lyon"]
    assert list(aav["LIBAAV2020"]) == ["Ajaccio", "Lyon", "Bourg-en-Bresse", "Paris"]
    assert msa["msa"].iloc[0].startswith("New York-Newark")
    assert not msa["msa"].str.contains("Division").any()


def test_download_when_files_are_missing(tmp_path, monkeypatch):
    brut = communes_brutes()
    contenu = io.BytesIO()
    with zipfile.ZipFile(contenu, "w") as z:
        z.writestr("donnees_communes.csv", brut.to_csv(sep=";", index=False))
    appels = []

    def faux_telechargement(url):
        appels.append(url)
        return contenu.getvalue()

    monkeypatch.setattr(data, "_telecharge", faux_telechargement)
    lu = data.lit_communes(tmp_path)
    assert appels == [data.URL_COMMUNES]
    assert len(lu) == len(brut)
    data.lit_communes(tmp_path)  # deuxième appel : lu sur disque
    assert len(appels) == 1
