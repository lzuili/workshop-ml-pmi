import numpy as np
import pandas as pd
import streamlit as st

from utils import data as D
from utils import plots as P
from utils import ui

ui.page_header(
    "3. Préparation des données",
    [
        "Repérer et traiter les valeurs manquantes.",
        "Ne pas confondre une valeur aberrante avec un événement rare… qu'on veut justement prévoir.",
        "Mettre les variables à la même échelle, et dire au modèle que 23h et minuit sont voisins.",
    ],
)

st.markdown(
    "On quitte le café pour de **vraies données** : OMNI (NASA), une mesure par heure de 2015 à 2024.  \n"
    "Ce sont les données du fil rouge : prédire l'activité géomagnétique (Kp) à partir du vent solaire."
)
st.markdown(
    "- **V** : vitesse du vent solaire (km/s).\n"
    "- **Bz** : composante nord-sud du champ magnétique du vent solaire (nT). Très négatif = propice aux orages.\n"
    "- **N** : densité du vent solaire (particules par cm³).\n"
    "- **Kp** : activité géomagnétique, de 0 (calme) à 9 (orage extrême).\n"
    "- **Dst** : perturbation du champ magnétique terrestre (nT), très négatif pendant un orage.\n"
    "- **F10.7** : activité du Soleil (flux radio)."
)

raw = ui.omni_raw()


@st.cache_data
def clean():
    return D.clean_omni(raw)


omni = clean()

# ------------------------------------------------------------
st.header("3.1 Valeurs manquantes")
st.markdown(
    "Quand l'instrument ne mesure rien, OMNI écrit une **valeur de remplissage** (9999 pour V, 999.9 pour N).  \n"
    "Si on la laisse, le modèle croit que le vent solaire a soufflé à 9999 km/s !"
)

GAP_START, GAP_END = "2023-02-01", "2023-02-28 23:00"
controls, chart = ui.demo_columns()
with controls:
    col = st.radio("Variable", ["v", "n"], format_func=lambda c: {"v": "V (vitesse)", "n": "N (densité)"}[c])
    mode = st.radio("Affichage", ["brut", "nettoyé"], horizontal=True)
    fill = D.FILL_VALUES[col]
    window_raw = raw[raw["time"].between(GAP_START, GAP_END)]
    n_missing = int((window_raw[col] >= fill).sum())
    st.metric("Heures manquantes (février 2023)", f"{n_missing} / {len(window_raw)}")
with chart:
    if mode == "brut":
        window = window_raw
        highlight = window[col] >= fill
    else:
        window = omni[omni["time"].between(GAP_START, GAP_END)]
        highlight = None
    unit = {"v": "V (km/s)", "n": "N (cm⁻³)"}[col]
    ui.plot(P.time_series(window, col, unit, highlight=highlight, highlight_name="valeur de remplissage",
                              name=unit.split(" ")[0]))
st.markdown(
    "Trois stratégies classiques :\n"
    "- **supprimer** les heures concernées (simple, mais on perd des données) ;\n"
    "- **interpoler** les petits trous (quelques heures) à partir des voisins ;\n"
    "- **garder une colonne « donnée manquante »** (0 ou 1) pour que le modèle le sache."
)
st.caption(
    "Rappel : pour le café, la valeur manquante du jour 7 était justement celle qu'on voulait prédire. "
    "Une cible manquante ne sert pas à l'entraînement."
)

with ui.aller_plus_loin():
    st.markdown(
        "Dans ce workshop (`utils/data.py`) : les valeurs de remplissage deviennent `NaN`, "
        "les trous de 3 h ou moins sont interpolés linéairement, et les heures restantes sans donnée "
        "sont retirées. Sur 2015 à 2024, cela ne concerne qu'environ 1 % des heures."
    )

# ------------------------------------------------------------
st.header("3.2 Valeur aberrante ou événement ?")
st.markdown(
    "Une méthode courante pour repérer les « outliers » : le **z-score**, c'est-à-dire à combien d'écarts-types "
    "de la moyenne se trouve un point.  \nAppliquons-la à l'indice Dst d'avril et mai 2024."
)

controls, chart = ui.demo_columns()
with controls:
    threshold = st.slider("Seuil de z-score", 1.0, 5.0, 3.0, step=0.5)
    storm = omni[omni["time"].between("2024-04-01", "2024-05-31 23:00")]
    z = (storm["dst"] - storm["dst"].mean()) / storm["dst"].std()
    outliers = z.abs() > threshold
    st.metric("Points détectés", int(outliers.sum()))
    if outliers.any():
        st.caption(f"Du {storm['time'][outliers].min():%d/%m %Hh} au {storm['time'][outliers].max():%d/%m %Hh}")
with chart:
    ui.plot(P.time_series(storm, "dst", "Dst (nT)", highlight=outliers, highlight_name="« outliers » détectés",
                              name="Dst"))
st.markdown(
    "**Constat** : les « outliers » détectés sont l'**orage géomagnétique du 10 au 11 mai 2024**, "
    "le plus fort depuis 20 ans. C'est exactement ce qu'on veut prévoir !"
)
ui.intuition(
    "en météo spatiale, les valeurs extrêmes sont souvent le signal. On ne supprime un point que s'il est "
    "physiquement impossible ou dû à une erreur d'instrument."
)

with ui.aller_plus_loin():
    st.latex(r"z = \frac{x - \bar{x}}{\sigma}")
    st.markdown("Un z-score de 3 signifie « à 3 écarts-types de la moyenne ».")

# ------------------------------------------------------------
st.header("3.3 Normalisation")
st.markdown(
    "V varie entre 300 et 800 km/s, Bz entre -20 et 20 nT. Sans précaution, le modèle « voit » surtout V.  \n"
    "On ramène donc chaque variable à une échelle comparable."
)


@st.cache_data
def train_solar_wind():
    train = D.fill_small_gaps(omni)
    train = train[train["time"].dt.year.between(*D.SPLIT_YEARS["train"])]
    return train[["v", "bz"]].dropna()


train_sw = train_solar_wind()
controls, chart = ui.demo_columns()
with controls:
    method = st.radio("Méthode", ["brut", "standardisation", "min-max"])
    if method == "brut":
        scaled = train_sw
        x_title = "valeur brute"
    elif method == "standardisation":
        scaled = (train_sw - train_sw.mean()) / train_sw.std()
        x_title = "valeur standardisée (moyenne 0, écart-type 1)"
    else:
        scaled = (train_sw - train_sw.min()) / (train_sw.max() - train_sw.min())
        x_title = "valeur min-max (entre 0 et 1)"
    st.dataframe(
        pd.DataFrame({"moyenne": scaled.mean(), "écart-type": scaled.std()}, index=["v", "bz"])
        .rename(index={"v": "V", "bz": "Bz"}).round(2)
    )
with chart:
    ui.plot(P.histograms(scaled, ["v", "bz"], ["V", "Bz"], x_title))
st.markdown(
    "Même principe qu'au module 1.5 : sans normalisation, la vallée de l'erreur est allongée "
    "et la descente de gradient peine."
)
st.warning(
    "⚠️ **Règle d'or** : la moyenne et l'écart-type (ou le min et le max) sont calculés **sur le train uniquement**, "
    "puis appliqués tels quels à la validation et au test. Sinon, on fait fuiter de l'information du test vers "
    "l'entraînement."
)

with ui.aller_plus_loin():
    st.latex(r"\text{standardisation : } x' = \frac{x - \bar{x}_{\text{train}}}{\sigma_{\text{train}}}"
             r"\qquad \text{min-max : } x' = \frac{x - \min_{\text{train}}}{\max_{\text{train}} - \min_{\text{train}}}")
    st.markdown(
        "Min-max est sensible aux valeurs extrêmes : un seul Bz à -42 nT écrase toutes les autres valeurs "
        "autour de 0.6 (voir l'histogramme). La standardisation est le choix par défaut le plus robuste."
    )

# ------------------------------------------------------------
st.header("3.4 Variables cycliques (encodage cos / sin)")
st.markdown(
    "L'heure va de 0 à 23. En valeur brute, 23h et minuit sont à une distance de 23 : pour le modèle, "
    "ils sont aux antipodes.  \nPlaçons plutôt les heures sur un cercle, comme sur une horloge."
)

tab_hour, tab_day = st.tabs(["Heure", "Jour de l'année"])
with tab_hour:
    controls, chart = ui.demo_columns()
    with controls:
        h1 = st.slider("Première heure", 0, 23, 23)
        h2 = st.slider("Seconde heure", 0, 23, 0)
        cos1, sin1 = D.cyclic_encode(h1, 24)
        st.caption(f"{h1}h devient (cos, sin) = ({cos1:.2f}, {sin1:.2f})")
    with chart:
        left, right = st.columns(2)
        with left:
            ui.plot(P.hours_on_line(h1, h2))
        with right:
            ui.plot(P.values_on_circle(h1, h2, 24, {0: "0h", 6: "6h", 12: "12h", 18: "18h"},
                                       "Sur un cercle (cos, sin)"))
with tab_day:
    controls, chart = ui.demo_columns()
    with controls:
        d1 = st.slider("Premier jour de l'année", 1, 365, 365)
        d2 = st.slider("Second jour de l'année", 1, 365, 1)
        st.caption(f"Jour {d1} : {pd.Timestamp(2023, 1, 1) + pd.Timedelta(days=d1 - 1):%d/%m}, "
                   f"jour {d2} : {pd.Timestamp(2023, 1, 1) + pd.Timedelta(days=d2 - 1):%d/%m}")
    with chart:
        st.markdown(f"Distance brute : **{abs(d1 - d2)} jours**")
        ui.plot(P.values_on_circle(d1, d2, 365.25, {1: "1er janv.", 91: "avril", 182: "juillet", 274: "octobre"},
                                   "Jours de l'année sur un cercle"))

ui.intuition("pour le modèle, 23h et minuit doivent être voisins. Le cercle le lui dit, la valeur brute non.")

with ui.aller_plus_loin():
    st.latex(r"\text{heure} \mapsto \left(\cos\frac{2\pi\,\text{heure}}{24},\ \sin\frac{2\pi\,\text{heure}}{24}\right)")
    st.markdown("Il faut les deux : avec le cosinus seul, 6h et 18h auraient la même valeur.")

ui.a_retenir(
    [
        "Remplacer les valeurs de remplissage par des NaN, puis décider : supprimer, interpoler ou signaler.",
        "Une valeur extrême n'est pas forcément une erreur : c'est souvent l'événement qu'on veut prévoir.",
        "Normaliser avec les statistiques du train seul ; encoder les variables cycliques en (cos, sin).",
    ]
)
