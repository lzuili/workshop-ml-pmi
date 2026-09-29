"""Récupère les données OMNI horaires (2015 à 2024) depuis CDAWeb et écrit un CSV brut.

À lancer une seule fois, avec internet :
    uv run --group data python data/fetch_omni.py

Le CSV garde les valeurs de remplissage OMNI (ex. 9999., 999.9) : le module 3
s'en sert pour illustrer les valeurs manquantes. Le nettoyage est dans utils/data.py.
"""

from pathlib import Path

import pandas as pd
from cdasws import CdasWs

DATASET = "OMNI2_H0_MRG1HR"
YEARS = range(2015, 2025)
OUTPUT = Path(__file__).parent / "omni_2015_2024_1h.csv"

# Nom CDAWeb -> nom de colonne dans le CSV
VARIABLES = {
    "KP1800": "kp10",  # Kp x 10 (divisé par 10 au nettoyage)
    "DST1800": "dst",  # indice Dst (nT)
    "F10_INDEX1800": "f107",  # flux radio solaire F10.7
    "V1800": "v",  # vitesse du vent solaire (km/s)
    "BZ_GSM1800": "bz",  # composante Bz du champ magnétique interplanétaire (nT)
    "N1800": "n",  # densité du vent solaire (cm^-3)
}


def fetch_year(cdas: CdasWs, year: int) -> pd.DataFrame:
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"
    status, data = cdas.get_data(DATASET, list(VARIABLES), start, end)
    if status["http"]["status_code"] != 200 or data is None:
        raise RuntimeError(f"Échec de la requête CDAWeb pour {year} : {status}")

    # cdasws remplace les valeurs de remplissage par NaN : on remet la valeur OMNI d'origine
    # (attribut FILLVAL) pour garder un fichier brut
    df = pd.DataFrame(
        {col: pd.Series(data[var].values).fillna(data[var].attrs["FILLVAL"]) for var, col in VARIABLES.items()}
    )
    # Les variables *1800 sont datées au milieu de l'heure (hh:30) : on garde le début d'heure
    df.insert(0, "time", pd.to_datetime(data["Epoch_1800"].values).floor("h"))
    return df


def main() -> None:
    cdas = CdasWs()
    frames = []
    for year in YEARS:
        df = fetch_year(cdas, year)
        print(f"{year} : {len(df)} heures")
        frames.append(df)

    omni = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    omni.to_csv(OUTPUT, index=False)
    print(f"Écrit : {OUTPUT} ({len(omni)} lignes)")


if __name__ == "__main__":
    main()
