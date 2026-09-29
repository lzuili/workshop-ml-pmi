"""Données du workshop : jeu café, données synthétiques et données OMNI.

Fonctions pures (sans Streamlit) : l'app et le notebook les importent toutes les deux
pour obtenir exactement les mêmes résultats.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OMNI_CSV = Path(__file__).resolve().parent.parent / "data" / "omni_2015_2024_1h.csv"

# ============================================================
# 1. Jeu « café » (modules 1 et 2)
# ============================================================

COFFEE_UNKNOWN_DAY = 7


def coffee_data() -> pd.DataFrame:
    """Ventes de café selon la température. Le jour 7 est inconnu (à prédire)."""
    return pd.DataFrame(
        {
            "jour": range(1, 13),
            "temperature": [12, 10, 12, 15, 18, 20, 22, 28, 25, 38, 30, 35],
            "ventes": [224, 218, 228, 250, 286, 303, np.nan, 368, 351, 425, 390, 407],
        }
    )


def coffee_xy() -> tuple[np.ndarray, np.ndarray]:
    """Températures et ventes des jours connus (sans le jour 7)."""
    df = coffee_data().dropna()
    return df["temperature"].to_numpy(float), df["ventes"].to_numpy(float)


def least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Droite des moindres carrés y = a*x + b."""
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


# ============================================================
# 2. Données synthétiques
# ============================================================


# Graine choisie pour que la courbe d'erreur de validation ait une forme en U bien lisible
CURVE_SEED = 164


def noisy_curve_data(n: int = 30, noise: float = 0.35, seed: int = CURVE_SEED) -> dict[str, np.ndarray]:
    """Points 1D bruités autour d'une courbe, séparés en train (2/3) et validation (1/3).

    Les points de validation sont pris à l'intérieur de la plage du train (pas d'extrapolation).
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = true_curve(x) + rng.normal(0, noise, n)
    is_val = np.zeros(n, dtype=bool)
    is_val[1 + rng.choice(n - 2, n // 3, replace=False)] = True
    return {"x_train": x[~is_val], "y_train": y[~is_val], "x_val": x[is_val], "y_val": y[is_val]}


def true_curve(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x) + 0.5 * x


def synthetic_predictions(
    noise: float, outlier: float, n: int = 120, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Valeurs vraies (signal journalier) et prédictions d'un modèle imparfait.

    noise : écart-type de l'erreur du modèle. outlier : taille d'une grosse erreur isolée.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y_true = 3 + 1.5 * np.sin(2 * np.pi * t / 24) + 0.4 * rng.normal(size=n).cumsum() / np.sqrt(n) * 4
    y_pred = y_true + noise * rng.normal(size=n)
    y_pred[n // 2] += outlier
    return y_true, y_pred


def daily_signal(n: int = 300, noise: float = 0.8, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Variation journalière bruitée (façon TEC) : pic l'après-midi, creux la nuit."""
    rng = np.random.default_rng(seed)
    hour = np.sort(rng.uniform(0, 24, n))
    y = daily_curve(hour) + rng.normal(0, noise, n)
    return hour, y


def daily_curve(hour: np.ndarray) -> np.ndarray:
    return 5 + 12 * np.exp(-(((hour - 14) / 3.5) ** 2)) + 3 * np.exp(-(((hour - 21) / 1.2) ** 2))


def synthetic_tec_map(n_lat: int = 60, n_lon: int = 120, seed: int = SEED) -> np.ndarray:
    """Image synthétique type carte de TEC (lignes = latitudes, colonnes = longitudes).

    Fond qui varie avec la latitude (deux crêtes équatoriales), un côté jour / nuit
    séparé par une frontière nette, et quelques structures localisées.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-90, 90, n_lat)[:, None]
    lon = np.linspace(-180, 180, n_lon)[None, :]
    crests = np.exp(-(((lat - 15) / 10) ** 2)) + np.exp(-(((lat + 15) / 10) ** 2))
    background = 10 + 25 * crests + 5 * np.cos(np.radians(lat))
    day = 1 / (1 + np.exp(-(lon - 20) / 4))  # frontière jour / nuit vers 20° de longitude
    tec = background * (0.35 + 0.65 * day)
    for lat0, lon0, amp in [(-50, -100, 18), (40, -60, 15), (60, 110, 20)]:
        tec += amp * np.exp(-(((lat - lat0) / 6) ** 2) - ((lon - lon0) / 10) ** 2)
    tec += rng.normal(0, 0.8, tec.shape)
    return tec


# ============================================================
# 3. Données OMNI (modules 3 à 5)
# ============================================================

# Valeurs de remplissage OMNI (« donnée manquante »)
FILL_VALUES = {"kp10": 99, "dst": 99999, "f107": 999.9, "v": 9999.0, "bz": 999.9, "n": 999.9}

VARIABLE_LABELS = {
    "kp": "Kp (activité géomagnétique, 0 à 9)",
    "dst": "Dst (nT, très négatif pendant un orage)",
    "f107": "F10.7 (activité solaire)",
    "v": "V : vitesse du vent solaire (km/s)",
    "bz": "Bz : champ magnétique interplanétaire (nT)",
    "n": "N : densité du vent solaire (cm⁻³)",
}

FEATURES = ["v", "bz", "n", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
TARGET = "kp_next"
STORM_TARGET = "storm_next"
STORM_KP = 5

SPLIT_YEARS = {"train": (2015, 2021), "val": (2022, 2022), "test": (2023, 2024)}


def load_omni_raw(path: Path = OMNI_CSV) -> pd.DataFrame:
    """CSV brut, avec les valeurs de remplissage OMNI."""
    return pd.read_csv(path, parse_dates=["time"])


def clean_omni(raw: pd.DataFrame) -> pd.DataFrame:
    """Remplace les valeurs de remplissage par NaN et convertit Kp (stocké x 10)."""
    df = raw.copy()
    for col, fill in FILL_VALUES.items():
        df[col] = df[col].where(df[col] < fill)
    df["kp"] = df.pop("kp10") / 10
    return df


def fill_small_gaps(df: pd.DataFrame, cols=("v", "bz", "n"), max_hours: int = 3) -> pd.DataFrame:
    """Interpole les petits trous (au plus max_hours heures). Les grands trous restent NaN."""
    df = df.copy()
    for col in cols:
        is_nan = df[col].isna()
        gap_id = (~is_nan).cumsum()
        gap_len = is_nan.groupby(gap_id).transform("sum")
        interpolated = df[col].interpolate(limit_area="inside")
        df[col] = df[col].where(~(is_nan & (gap_len <= max_hours)), interpolated)
    return df


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode l'heure et le jour de l'année sur un cercle (cos, sin)."""
    df = df.copy()
    hour = df["time"].dt.hour
    doy = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def cyclic_encode(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.cos(angle)), float(np.sin(angle))


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Cible : Kp de l'heure suivante, et « orage » (Kp >= 5) à l'heure suivante."""
    df = df.copy()
    df[TARGET] = df["kp"].shift(-1)
    df[STORM_TARGET] = (df[TARGET] >= STORM_KP).astype(float).where(df[TARGET].notna())
    return df


def load_dataset() -> pd.DataFrame:
    """Données OMNI prêtes pour le ML : nettoyées, petits trous comblés, features et cibles.

    Les lignes où il manque une feature ou la cible (grands trous) sont retirées.
    """
    df = clean_omni(load_omni_raw())
    df = fill_small_gaps(df)
    df = add_cyclic_features(df)
    df = add_targets(df)
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def split_labels_temporal(df: pd.DataFrame) -> pd.Series:
    """'train' / 'val' / 'test' selon l'année (split temporel de référence)."""
    year = df["time"].dt.year
    labels = pd.Series("test", index=df.index)
    for name, (start, end) in SPLIT_YEARS.items():
        labels[year.between(start, end)] = name
    return labels


def split_labels_random(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Même proportions que le split temporel, mais heures tirées au hasard."""
    temporal = split_labels_temporal(df)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.permutation(temporal.to_numpy()), index=df.index)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = split_labels_temporal(df)
    return df[labels == "train"], df[labels == "val"], df[labels == "test"]


def fit_scaler(train: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """Moyenne et écart-type calculés sur le train uniquement."""
    return train[cols].mean(), train[cols].std()


def apply_scaler(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return (df[mean.index] - mean) / std
