import numpy as np
import pandas as pd
import streamlit as st

from utils import data as D
from utils import models as M
from utils import plots as P
from utils import ui

ui.page_header(
    "2. Notions clés",
    [
        "Le vocabulaire du ML, sur l'exemple du café puis sur la météo spatiale.",
        "La différence entre régression et classification.",
        "Pourquoi un modèle trop simple ou trop complexe se trompe sur de nouvelles données.",
    ],
)

# ------------------------------------------------------------
st.header("2.1 Vocabulaire")
st.markdown("Reprenons le café : chaque mot du ML y a déjà un équivalent.")
st.image("assets/cafe_features_labels.png", width=900)
st.caption("La présentation d'origine utilisait un jeu un peu différent, d'où b = 139.09 au lieu de 138.14 ici.")

vocabulary = pd.DataFrame(
    [
        ("Inputs / features", "Ce que l'on connaît et donne au modèle.", "température",
         "vitesse du vent solaire, Bz, heure"),
        ("Label / cible", "Ce que l'on veut prédire (connu pendant l'entraînement).", "ventes",
         "Kp de l'heure suivante"),
        ("Output / prédiction", "Ce que le modèle répond.", "ventes prédites", "Kp prédit"),
        ("Poids (paramètres)", "Les nombres que le modèle apprend.", "a, b", "des milliers de poids d'un réseau"),
        ("Modèle / architecture", "La forme du calcul, avant apprentissage.", "droite y = a·x + b",
         "MLP, CNN, Transformer (module 5)"),
        ("Loss (fonction d'erreur)", "Le nombre que l'entraînement cherche à rendre petit.", "MSE",
         "MSE, MAE, entropie croisée…"),
        ("Hyperparamètres", "Les réglages que nous choisissons avant l'entraînement.",
         "learning rate, nombre d'itérations", "+ nombre de couches, epochs, batch size"),
    ],
    columns=["Terme", "Définition", "Café", "Météo spatiale"],
)
st.table(vocabulary.set_index("Terme"))

c1, c2 = st.columns(2)
c1.markdown(
    "**Poids ≠ hyperparamètres**  \n"
    "Les **poids** sont appris par le modèle (a et b).  \n"
    "Les **hyperparamètres** sont choisis par nous (le learning rate)."
)
c2.markdown(
    "**Epoch et batch**  \n"
    "Une **epoch** = un passage complet sur toutes les données d'entraînement.  \n"
    "Un **batch** = le petit paquet d'exemples utilisé pour faire un pas de descente."
)

# ------------------------------------------------------------
st.header("2.2 Régression vs classification")
c1, c2 = st.columns(2)
c1.markdown("**Régression** : prédire un **nombre**.  \nExemple : Kp = 4.3 dans une heure.")
c2.markdown("**Classification** : prédire une **catégorie**.  \nExemple : orage ou pas orage dans une heure.")
ui.plot(P.regression_vs_classification())

# ------------------------------------------------------------
st.header("2.3 Démo : sous-apprentissage et sur-apprentissage")
st.markdown(
    "Des points bruités autour d'une courbe. On ajuste un polynôme sur les points **pleins** (train), "
    "et on regarde son erreur sur les points **creux** (validation), que le modèle n'a jamais vus.  \n"
    "Le degré du polynôme règle la complexité du modèle."
)


@st.cache_data
def curve_data():
    data = D.noisy_curve_data()
    return data, M.poly_errors(data, range(1, 16))


data, errors = curve_data()

controls, chart = ui.demo_columns()
with controls:
    degree = st.slider("Degré du polynôme", 1, 15, 1)
    row = errors.set_index("degre").loc[degree]
    st.metric("Erreur train", f"{row['train']:.3f}")
    st.metric("Erreur validation", f"{row['validation']:.3f}")
    best = int(errors.loc[errors["validation"].idxmin(), "degre"])
    if degree <= 2:
        st.warning("Sous-apprentissage : le modèle est trop simple, il rate la tendance.")
    elif row["validation"] > 2 * errors["validation"].min():
        st.error("Sur-apprentissage : le modèle colle au bruit du train et se trompe ailleurs.")
    else:
        st.success("Bon compromis : le modèle suit la tendance sans apprendre le bruit.")
with chart:
    left, right = st.columns(2)
    with left:
        ui.plot(P.poly_fit_plot(data, M.fit_poly(data["x_train"], data["y_train"], degree), degree,
                                curve=D.true_curve))
    with right:
        ui.plot(P.poly_error_plot(errors, degree))

ui.intuition(
    "trop simple, le modèle rate la tendance. Trop complexe, il apprend le bruit par cœur "
    "et se trompe sur de nouvelles données. On choisit donc le modèle sur l'erreur de **validation**, "
    f"pas sur celle du train (ici, le meilleur degré est {best})."
)

with ui.aller_plus_loin():
    st.markdown(
        "L'erreur de train baisse **toujours** quand la complexité augmente : un modèle assez flexible "
        "peut passer par tous les points. C'est pourquoi elle ne sert pas à choisir le modèle.  \n"
        "Un polynôme de degré $d$ a $d + 1$ poids. Avec 20 points de train, le degré 15 a presque "
        "autant de poids que de points : il a de quoi les apprendre par cœur."
    )

ui.a_retenir(
    [
        "Features = ce qu'on sait, label = ce qu'on veut prédire, poids = ce que le modèle apprend.",
        "Régression = un nombre ; classification = une catégorie.",
        "On choisit la complexité du modèle sur l'erreur de validation, jamais sur celle du train.",
    ]
)
