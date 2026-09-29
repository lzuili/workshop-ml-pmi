import numpy as np
import streamlit as st

from utils import data as D
from utils import models as M
from utils import plots as P
from utils import ui

ASSETS = "assets"

df = D.coffee_data()
x, y = D.coffee_xy()
A_BEST, B_BEST = D.least_squares(x, y)
DAY7_TEMP = float(df.loc[df["jour"] == D.COFFEE_UNKNOWN_DAY, "temperature"].iloc[0])

ui.page_header(
    "1. Introduction",
    [
        "Ce qui distingue le Machine Learning de la programmation classique.",
        "Apprendre = trouver les paramètres qui rendent l'erreur la plus petite possible.",
        "Comment la descente de gradient trouve ces paramètres, pas à pas.",
    ],
)

# ------------------------------------------------------------
st.header("1.1 ML vs programmation classique")
st.image(f"{ASSETS}/ml_vs_classique.png", width=900)
col1, col2 = st.columns(2)
col1.markdown("**Programmation classique : « fais ça »**  \nOn écrit nous-mêmes les règles, instruction par instruction.")
col2.markdown(
    "**Machine Learning : « dis-moi ceci à partir de cela »**  \n"
    "On donne des exemples, le modèle apprend les règles tout seul."
)

# ------------------------------------------------------------
st.header("1.2 L'exemple du café")
st.markdown(
    "Le Great Coffee House note chaque jour la température extérieure et le nombre de cafés vendus.  \n"
    f"Le jour {D.COFFEE_UNKNOWN_DAY}, la vente n'a pas été notée. **Combien de cafés ce jour-là ?**"
)
c_table, c_plot = st.columns([1, 2], gap="large")
c_table.dataframe(
    df.assign(ventes=df["ventes"].map(lambda v: "?" if np.isnan(v) else f"{v:.0f}")).rename(columns={"jour": "Jour", "temperature": "Température (°C)", "ventes": "Ventes (tasses)"}),
    hide_index=True, height=460,
)
with c_plot:
    ui.plot(P.coffee_plot(df))

# ------------------------------------------------------------
st.header("1.3 Démo : trouver a et b à la main")
st.markdown(
    "Tentons une droite : **ventes = a × température + b**.  \n"
    "Bougez `a` et `b`. Les pointillés rouges sont les erreurs de la droite sur chaque jour connu."
)

st.session_state.setdefault("coffee_a", 3.0)
st.session_state.setdefault("coffee_b", 200.0)


def show_best_line():
    st.session_state["coffee_a"] = round(A_BEST, 2)
    st.session_state["coffee_b"] = round(B_BEST, 1)


controls, chart = ui.demo_columns()
with controls:
    a = st.slider("a (pente)", -5.0, 15.0, step=0.05, key="coffee_a")
    b = st.slider("b (ordonnée à l'origine)", 0.0, 300.0, step=0.5, key="coffee_b")
    st.button("Montrer la meilleure droite", on_click=show_best_line, type="primary")
    error = M.mse(y, a * x + b)
    st.metric("Erreur (MSE)", f"{error:,.0f}".replace(",", " "))
    st.metric("Erreur typique (RMSE)", f"{np.sqrt(error):.1f} tasses")
    st.metric(f"Jour {D.COFFEE_UNKNOWN_DAY} prédit", f"{a * DAY7_TEMP + b:.0f} tasses")
with chart:
    ui.plot(P.coffee_plot(df, line=(a, b), residuals=True))

ui.intuition("apprendre, c'est trouver les valeurs de a et b qui rendent l'erreur la plus petite possible.")

with ui.aller_plus_loin():
    st.markdown("L'erreur quadratique moyenne (MSE) sur les $n$ jours connus :")
    st.latex(r"\text{MSE}(a, b) = \frac{1}{n}\sum_{i=1}^{n} \big(y_i - (a\,x_i + b)\big)^2")
    st.markdown(
        "On met les erreurs au carré pour que les erreurs positives et négatives ne se compensent pas, "
        "et pour pénaliser davantage les grosses erreurs. La RMSE est la racine de la MSE : "
        "elle s'exprime dans l'unité de la cible (ici, des tasses).  \n"
        f"Meilleure droite (moindres carrés) : a = {A_BEST:.3f}, b = {B_BEST:.2f}."
    )

# ------------------------------------------------------------
st.header("1.4 Démo : le paysage de l'erreur")
st.markdown(
    "Chaque couple (a, b) donne une erreur. En colorant toutes les combinaisons, on obtient une carte.  \n"
    "Le point rouge suit vos sliders de la section 1.3. L'étoile marque le minimum."
)


@st.cache_data
def raw_valley():
    a_values = np.linspace(-5, 15, 161)
    b_values = np.linspace(-50, 300, 141)
    return a_values, b_values, M.mse_grid(x, y, a_values, b_values)


a_grid, b_grid, mse_raw = raw_valley()
_, chart = st.columns([1, 3], gap="large")
with chart:
    ui.plot(P.valley_plot(a_grid, b_grid, mse_raw, minimum=(A_BEST, B_BEST), current=(a, b)))
ui.intuition(
    "chaque point de la carte est un modèle possible. Entraîner, c'est descendre vers le fond de la vallée."
)

# ------------------------------------------------------------
st.header("1.5 Démo : la descente de gradient")
st.markdown(
    "L'ordinateur ne voit pas la carte : il ne connaît que la pente là où il se trouve (le **gradient**).  \n"
    "À chaque pas, il fait un petit pas dans la direction qui descend. La taille du pas est le **learning rate**."
)

LEARNING_RATES = [0.0001, 0.0003, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9, 1.1]


def new_descent(normalize: bool) -> M.LineGradientDescent:
    return M.LineGradientDescent(x, y, normalize=normalize)


controls, chart = ui.demo_columns()
with controls:
    normalize = st.checkbox("Normaliser la température", value=True)
    lr = st.select_slider("Learning rate (échelle log)", options=LEARNING_RATES, value=0.1)
    n_steps = st.slider("Nombre de pas pour « N pas »", 1, 200, 20)

    descent = st.session_state.get("descent")
    if descent is None or descent.normalize != normalize:
        descent = st.session_state["descent"] = new_descent(normalize)

    b1, b2 = st.columns(2)
    if b1.button("1 pas"):
        descent.step(lr, 1)
    if b2.button("N pas"):
        descent.step(lr, n_steps)
    if st.button("Réinitialiser"):
        descent = st.session_state["descent"] = new_descent(normalize)

    a_cur, b_cur = descent.to_original(descent.params[-1])
    st.metric("Pas effectués", len(descent.params) - 1)
    if descent.diverged:
        st.error("La descente a divergé : le pas est trop grand, on saute par-dessus la vallée.")
    else:
        st.metric("Erreur actuelle (MSE)", f"{descent.losses[-1]:,.0f}".replace(",", " "))
        st.caption(f"Droite actuelle : a = {a_cur:.2f}, b = {b_cur:.1f}")


@st.cache_data
def normalized_valley():
    mean, std = x.mean(), x.std()
    a_values = np.linspace(-150, 300, 181)
    b_values = np.linspace(-50, 400, 181)
    return a_values, b_values, M.mse_grid((x - mean) / std, y, a_values, b_values)


with chart:
    if normalize:
        a_values, b_values, mse_values = normalized_valley()
        best = (A_BEST * x.std(), A_BEST * x.mean() + B_BEST)
        titles = ("a' (pente, température normalisée)", "b'")
        st.caption("Paysage de l'erreur vu par la descente, avec la température normalisée : la vallée est ronde.")
    else:
        a_values, b_values, mse_values = a_grid, b_grid, mse_raw
        best = (A_BEST, B_BEST)
        titles = ("a (pente)", "b (ordonnée à l'origine)")
        st.caption("Paysage de l'erreur avec la température brute : la vallée est très allongée.")
    trajectory = descent.trajectory()
    if descent.diverged:
        trajectory = trajectory[:-1]
    left, right = st.columns([3, 2])
    with left:
        ui.plot(P.valley_plot(a_values, b_values, mse_values, minimum=best, trajectory=trajectory,
                              axis_titles=titles, equal_axes=normalize))
    with right:
        ui.plot(P.loss_curve(descent.losses, title="Erreur au fil des pas", height=480))

st.markdown(
    "**Essayez** : avec la température normalisée, un learning rate de 0.1 à 0.9 atteint le fond en quelques pas ; "
    "au-delà de 1, ça diverge.  \n"
    "Décochez la normalisation : dès 0.002 ça diverge, et en dessous la descente met des milliers de pas "
    "à longer la vallée. On y revient au module 3 (normalisation)."
)
ui.intuition(
    "le gradient indique la pente, le learning rate la taille du pas. "
    "Trop petit : c'est lent. Trop grand : on saute par-dessus la vallée."
)

with ui.aller_plus_loin():
    st.markdown("Le gradient de la MSE par rapport à chaque paramètre, puis la mise à jour (η = learning rate) :")
    st.latex(r"\frac{\partial \text{MSE}}{\partial a} = \frac{2}{n}\sum_i (a x_i + b - y_i)\,x_i \qquad "
             r"\frac{\partial \text{MSE}}{\partial b} = \frac{2}{n}\sum_i (a x_i + b - y_i)")
    st.latex(r"a \leftarrow a - \eta\,\frac{\partial \text{MSE}}{\partial a} \qquad "
             r"b \leftarrow b - \eta\,\frac{\partial \text{MSE}}{\partial b}")
    st.markdown(
        "Normaliser : $x' = (x - \\bar{x}) / \\sigma_x$. La droite devient $y = a' x' + b'$, "
        "avec $a = a'/\\sigma_x$ et $b = b' - a'\\bar{x}/\\sigma_x$ : c'est le même modèle, écrit autrement. "
        "Mais la vallée devient ronde, et un même pas convient aux deux paramètres."
    )

# ------------------------------------------------------------
st.header("1.6 Synthèse")
st.image(f"{ASSETS}/synthese_ml.png", width=900)
st.markdown(
    "**Des données** (ce qu'on sait : les features ; ce qu'on veut prédire : les labels) "
    "**+ un modèle** → on **minimise l'erreur** entre labels et prédictions → on obtient **les poids du modèle**."
)

ui.a_retenir(
    [
        "Le ML apprend les règles à partir d'exemples, au lieu qu'on les écrive.",
        "Entraîner = chercher les paramètres qui minimisent l'erreur (la loss).",
        "La descente de gradient y arrive pas à pas ; le learning rate règle la taille des pas.",
    ]
)
