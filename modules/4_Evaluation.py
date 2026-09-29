import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)

from utils import data as D
from utils import models as M
from utils import plots as P
from utils import ui

ui.page_header(
    "4. Évaluer un modèle",
    [
        "Pourquoi on sépare les données en train, validation et test, et comment le faire avec des séries temporelles.",
        "Lire les métriques de régression : MAE, RMSE, R².",
        "Pourquoi l'accuracy est trompeuse quand les orages sont rares, et comment régler le seuil de décision.",
    ],
)

dataset = ui.omni_dataset()

# ------------------------------------------------------------
st.header("4.1 Train, validation, test")
c1, c2, c3 = st.columns(3)
c1.markdown("**Train = les exercices**  \nLe modèle apprend dessus.")
c2.markdown("**Validation = l'examen blanc**  \nPour choisir les hyperparamètres et comparer les modèles.")
c3.markdown("**Test = l'examen final**  \nUtilisé **une seule fois**, à la fin, pour annoncer le score.")


@st.cache_data(show_spinner="Entraînement des deux random forests (une seule fois)…")
def split_scores():
    """Même modèle, entraîné et testé avec les deux façons de découper."""
    scores = {}
    for name, labels in [("temporel", D.split_labels_temporal(dataset)), ("aléatoire", D.split_labels_random(dataset))]:
        train, test = dataset[labels == "train"], dataset[labels == "test"]
        model = M.train_forest(train[D.FEATURES], train[D.TARGET])
        pred = model.predict(test[D.FEATURES])
        scores[name] = {"R²": r2_score(test[D.TARGET], pred), "MAE": mean_absolute_error(test[D.TARGET], pred)}
    return scores


@st.cache_data
def split_labels(kind: str) -> pd.Series:
    return D.split_labels_temporal(dataset) if kind == "temporel" else D.split_labels_random(dataset)


controls, chart = ui.demo_columns()
with controls:
    kind = st.radio("Découpage", ["temporel", "aléatoire"], format_func=lambda k: f"split {k}")
    scores = split_scores()
    st.metric("R² sur le test (random forest)", f"{scores[kind]['R²']:.2f}",
              delta=None if kind == "temporel" else f"{scores[kind]['R²'] - scores['temporel']['R²']:+.2f} vs temporel",
              delta_color="off")
    st.metric("MAE sur le test (en Kp)", f"{scores[kind]['MAE']:.2f}")
labels = split_labels(kind)
with chart:
    daily = dataset["time"].dt.hour == 12  # une heure par jour suffit pour la frise
    st.markdown("**Répartition des heures, 2015 à 2024**")
    ui.plot(P.split_timeline(dataset["time"][daily], labels[daily]))
    zoom = dataset["time"].between("2023-04-22", "2023-04-26")
    st.markdown("**Zoom sur 4 jours (orage du 23 au 24 avril 2023) : Kp de l'heure suivante, la cible**")
    ui.plot(P.zoom_split(dataset[zoom], labels[zoom], D.TARGET, "Kp (heure suivante)"))

if kind == "aléatoire":
    st.error(
        "Avec un split aléatoire, chaque heure de test est entourée d'heures de train presque identiques "
        "(Kp est même constant par blocs de 3 h). Le modèle « recopie » ses voisins : le score est trop optimiste."
    )
ui.intuition(
    "avec des séries temporelles, l'heure suivante ressemble beaucoup à l'heure précédente. "
    "Un split aléatoire laisse le modèle « tricher ». On découpe donc par blocs de temps : "
    "train 2015 à 2021, validation 2022, test 2023 à 2024."
)

with ui.aller_plus_loin():
    st.markdown(
        "Le modèle est une random forest scikit-learn (60 arbres), avec les features du fil rouge : "
        "V, Bz, N et l'heure et le jour de l'année en (cos, sin). Seul le découpage change.  \n"
        "Une partie de l'écart vient aussi du fait que 2023 et 2024 (maximum solaire) sont plus agités que "
        "2015 à 2021 : le split temporel mesure aussi la capacité à s'adapter à des conditions nouvelles, "
        "ce qui est exactement ce qu'on attend d'une prévision."
    )

# ------------------------------------------------------------
st.header("4.2 Métriques de régression : MAE, RMSE, R²")
st.markdown("Un modèle imaginaire prédit une valeur à chaque heure. Réglez sa qualité et observez les trois métriques.")

controls, chart = ui.demo_columns()
with controls:
    noise = st.slider("Bruit du modèle", 0.0, 1.5, 0.4, step=0.05)
    outlier = st.slider("Une grosse erreur isolée", 0.0, 10.0, 0.0, step=0.5)
    y_true, y_pred = D.synthetic_predictions(noise, outlier)
    st.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.2f}")
    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    st.metric("R²", f"{r2_score(y_true, y_pred):.2f}")
with chart:
    ui.plot(P.true_vs_pred(y_true, y_pred))
st.markdown(
    "- **MAE** : l'erreur moyenne, dans l'unité de la cible (ici, des points de Kp).\n"
    "- **RMSE** : pénalise davantage les grosses erreurs. Bougez le slider « grosse erreur » : la RMSE bondit, "
    "la MAE bouge peu.\n"
    "- **R²** : la part de la variabilité expliquée. 1 = parfait, 0 = pas mieux que prédire la moyenne, "
    "négatif = pire que la moyenne."
)

with ui.aller_plus_loin():
    st.latex(r"\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i| \qquad "
             r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2} \qquad "
             r"R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}")

# ------------------------------------------------------------
st.header("4.3 Métriques de classification : accuracy, précision, rappel")
st.markdown(
    "Tâche : prédire **orage (Kp ≥ 5) ou pas** dans l'heure qui suit, sur les données de test (2023 à 2024).  \n"
    "Une régression logistique donne une **probabilité d'orage** ; on annonce un orage si elle dépasse un **seuil**."
)


@st.cache_data(show_spinner="Entraînement de la régression logistique…")
def storm_probabilities():
    train, _, test = D.temporal_split(dataset)
    mean, std = D.fit_scaler(train, D.FEATURES)
    model = M.train_logreg(D.apply_scaler(train, mean, std), train[D.STORM_TARGET])
    proba = model.predict_proba(D.apply_scaler(test, mean, std))[:, 1]
    return test[["time", D.STORM_TARGET]].reset_index(drop=True), proba


test, proba = storm_probabilities()
y_true = test[D.STORM_TARGET].to_numpy().astype(int)

controls, chart = ui.demo_columns()
with controls:
    always_calm = st.toggle("Toujours « calme »", help="Un modèle qui prédit toujours calme, sans jamais annoncer d'orage.")
    threshold = st.slider("Seuil de décision", 0.02, 0.98, 0.5, step=0.02, disabled=always_calm)
    y_pred = np.zeros_like(y_true) if always_calm else (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.1%}")
    st.metric("Précision", f"{precision_score(y_true, y_pred):.1%}" if tp + fp else "non définie")
    st.metric("Rappel", f"{recall_score(y_true, y_pred):.1%}")
with chart:
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Matrice de confusion**")
        ui.plot(P.confusion_matrix_plot(tp, fp, fn, tn))
    with right:
        may = test["time"].between("2024-05-08", "2024-05-15")
        ui.plot(P.storm_probability_plot(test[may], proba[may.to_numpy()], None if always_calm else threshold))

if always_calm:
    st.error(
        f"Ce « modèle » ne prévoit jamais d'orage, et pourtant son accuracy est de {accuracy_score(y_true, y_pred):.1%} : "
        f"seules {y_true.mean():.1%} des heures de test sont des orages. Son rappel est de 0 : il les rate tous."
    )
st.markdown(
    "- **Précision** : « quand j'annonce un orage, ai-je raison ? »\n"
    "- **Rappel** : « parmi les vrais orages, combien en ai-je vu ? »"
)
ui.intuition(
    "quand une classe est rare, l'accuracy est trompeuse. Le seuil règle le compromis : "
    "le baisser donne moins d'orages manqués mais plus de fausses alertes."
)

with ui.aller_plus_loin():
    st.latex(r"\text{précision} = \frac{\text{orages annoncés et réels}}{\text{orages annoncés}} \qquad "
             r"\text{rappel} = \frac{\text{orages annoncés et réels}}{\text{orages réels}}")
    st.markdown(
        "Le bon seuil dépend de l'usage : pour un opérateur GNSS, manquer un orage coûte-t-il plus cher "
        "qu'une fausse alerte ? Le seuil se choisit sur la **validation**, jamais sur le test."
    )

ui.a_retenir(
    [
        "Train pour apprendre, validation pour choisir, test une seule fois à la fin.",
        "Séries temporelles : découper par blocs de temps, jamais au hasard.",
        "Classe rare : regarder précision et rappel, pas l'accuracy ; le seuil règle le compromis.",
    ]
)
