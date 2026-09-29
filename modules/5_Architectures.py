import numpy as np
import pandas as pd
import streamlit as st

from utils import data as D
from utils import models as M
from utils import plots as P
from utils import ui

ui.page_header(
    "5. Architectures",
    [
        "Un neurone, c'est la régression du module 1 plus une fonction d'activation.",
        "En empilant des neurones (MLP), on obtient des modèles très flexibles.",
        "Les CNN pour les cartes, les Transformers pour les séquences : l'intuition derrière chacun.",
    ],
)

# ------------------------------------------------------------
st.header("5.1 Le neurone")
st.markdown(
    "Un neurone fait exactement ce que faisait notre droite du café : **poids × entrée + biais**.  \n"
    "Puis il applique une **fonction d'activation**, qui tord la droite."
)

controls, chart = ui.demo_columns()
with controls:
    weight = st.slider("Poids", -3.0, 3.0, 1.0, step=0.1)
    bias = st.slider("Biais", -3.0, 3.0, 0.0, step=0.1)
    activation = st.radio("Activation", list(M.ACTIVATIONS), horizontal=True)
with chart:
    xs = np.linspace(-4, 4, 400)
    ui.plot(P.neuron_plot(xs, M.neuron(xs, weight, bias, activation), activation))
ui.intuition(
    "sans activation, empiler des neurones donne toujours une droite. "
    "L'activation (ReLU : « 0 si négatif », sigmoïde : « écrase entre 0 et 1 ») permet d'apprendre des courbes."
)

with ui.aller_plus_loin():
    st.latex(r"\text{sortie} = f(w\,x + b) \qquad \text{ReLU}(z) = \max(0, z) \qquad "
             r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    st.markdown("Avec plusieurs entrées : $f(w_1 x_1 + w_2 x_2 + \\dots + b)$, une somme pondérée.")

# ------------------------------------------------------------
st.header("5.2 Le MLP (réseau de neurones multicouche)")
st.markdown(
    "On range les neurones en **couches** : chaque neurone reçoit toutes les sorties de la couche précédente.  \n"
    "Tâche : retrouver une variation journalière bruitée (pic l'après-midi, rebond le soir)."
)


@st.cache_data
def daily_data():
    hour, y = D.daily_signal()
    rng = np.random.default_rng(D.SEED)
    is_val = np.zeros(len(hour), dtype=bool)
    is_val[rng.choice(len(hour), len(hour) // 3, replace=False)] = True
    return hour, y, is_val


@st.cache_resource(show_spinner=False)
def warm_up_torch():
    """Le tout premier entraînement PyTorch est lent (initialisation) : on le fait au chargement de la page."""
    M.train_mlp_1d(np.linspace(0, 1, 10), np.linspace(0, 1, 10), [2], epochs=5)
    return True


warm_up_torch()


@st.cache_data(show_spinner="Entraînement du réseau…")
def train(hidden: tuple[int, ...]):
    hour, y, is_val = daily_data()
    model, losses, val_losses, scaling = M.train_mlp_1d(
        hour[~is_val], y[~is_val], list(hidden), x_val=hour[is_val], y_val=y[is_val]
    )
    grid = np.linspace(0, 24, 300)
    return grid, M.predict_mlp_1d(model, grid, scaling), losses, val_losses


controls, chart = ui.demo_columns()
with controls:
    n_layers = st.slider("Couches cachées", 1, 3, 1)
    n_neurons = st.slider("Neurones par couche", 1, 32, 2)
    hidden = tuple([n_neurons] * n_layers)
    st.metric("Poids à apprendre", M.count_parameters(1, list(hidden)))
    if st.button("Entraîner", type="primary"):
        st.session_state["mlp_trained"] = hidden
    trained = st.session_state.get("mlp_trained") == hidden
with chart:
    ui.plot(P.network_diagram([1, *hidden, 1]))

hour, y, is_val = daily_data()
if trained:
    grid, pred, losses, val_losses = train(hidden)
    left, right = st.columns(2)
    with left:
        ui.plot(P.mlp_fit_plot(hour[~is_val], y[~is_val], grid, pred, true_curve=D.daily_curve(grid)))
    with right:
        fig = P.loss_curve(losses, val_losses=val_losses, name="train", title="Loss au fil des epochs", height=400)
        fig.update_xaxes(title="epoch")
        ui.plot(fig)
    st.caption(f"Loss finale : train {losses[-1]:.2f}, validation {val_losses[-1]:.2f} "
               "(le bruit des données vaut à lui seul environ 0.64).")
else:
    st.info("Choisissez une architecture puis cliquez sur **Entraîner** (300 epochs, moins d'une seconde).")

ui.intuition(
    "plus de neurones = plus de flexibilité, mais aussi plus de risque de sur-apprentissage (module 2.3) : "
    "surveillez l'écart entre la loss de train et celle de validation."
)
st.markdown("Le code complet de ce MLP (PyTorch), avec la boucle d'entraînement expliquée ligne par ligne, "
            "est dans le notebook.")

# ------------------------------------------------------------
st.header("5.3 Le CNN (réseau convolutif)")
st.markdown(
    "Pour une image ou une carte, un CNN utilise un **filtre** : une petite loupe de 3 × 3 pixels "
    "qui glisse sur l'image et réagit à un motif local.  \nChaque position de la loupe donne un pixel de la carte de sortie."
)


@st.cache_data
def tec_map():
    return D.synthetic_tec_map()


image = tec_map()
controls, chart = ui.demo_columns()
with controls:
    kernel_name = st.radio("Filtre", list(M.KERNELS))
    kernel = M.KERNELS[kernel_name]
    k = kernel.shape[0]
    row = st.slider("Position verticale du filtre", 0, image.shape[0] - k, 20)
    col = st.slider("Position horizontale du filtre", 0, image.shape[1] - k, 60)
    st.markdown("**Le filtre**")
    rows = r" \\ ".join(" & ".join("1/9" if kernel_name == "flou" else f"{v:g}" for v in line) for line in kernel)
    st.latex(r"\begin{pmatrix}" + rows + r"\end{pmatrix}")
    output = M.conv2d(image, kernel)
    st.metric("Pixel de sortie", f"{output[row, col]:.1f}")
    st.caption("= somme de (fenêtre × filtre), case par case")
with chart:
    zmid = None if kernel_name == "flou" else 0
    scale = "Viridis" if kernel_name == "flou" else P.DIVERGING
    ui.plot(P.image_plot(image, "Carte d'entrée (type TEC) et position du filtre", window=(row, col, k)))
    ui.plot(P.image_plot(output, f"Carte de sortie : {kernel_name}", pixel=(row, col),
                         colorscale=scale, zmid=zmid))
st.markdown(
    "Le flou lisse le bruit. Les contours horizontaux réagissent aux variations en latitude "
    "(les crêtes équatoriales) ; les contours verticaux à la frontière jour / nuit."
)
ui.intuition(
    "dans un CNN, les filtres ne sont pas choisis à la main : ils sont **appris**, comme les poids d'un MLP. "
    "Idéal pour des cartes (TEC, images)."
)

with ui.aller_plus_loin():
    st.latex(r"\text{sortie}[i, j] = \sum_{u=0}^{2}\sum_{v=0}^{2} \text{filtre}[u, v]\;\text{entrée}[i+u,\ j+v]")
    st.markdown("Un CNN empile plusieurs couches de filtres : les premières voient des bords, "
                "les suivantes des formes de plus en plus grandes.")

# ------------------------------------------------------------
st.header("5.4 Le Transformer (intuition)")
st.markdown(
    "Le cœur du Transformer est l'**attention** : pour produire une sortie, le modèle décide à quels éléments "
    "de l'entrée « prêter attention », puis en fait une moyenne pondérée.  \n"
    "Analogie : dans une phrase, pour comprendre « il », on regarde le nom auquel il se rapporte."
)
st.warning("⚠️ Illustration, pas un modèle entraîné : les poids ci-dessous sont calculés par une simple "
           "similarité entre heures (niveau de Dst et variation sur 3 h).")

WINDOW = 48


@st.cache_data
def dst_storm():
    omni = D.clean_omni(D.load_omni_raw())
    storm = omni[omni["time"].between("2024-05-07", "2024-05-15")].reset_index(drop=True)
    return storm, M.attention_representations(storm["dst"].to_numpy(float))


storm, reps = dst_storm()
controls, chart = ui.demo_columns()
with controls:
    query = st.slider("Instant à prédire", WINDOW, len(storm) - 1, WINDOW + 60,
                      format="%d h après le 07/05")
    st.caption(f"{storm['time'][query]:%d/%m à %Hh}, Dst = {storm['dst'][query]:.0f} nT")
    temperature = st.select_slider("Concentration de l'attention (température)",
                                   options=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0], value=1.0,
                                   help="Température basse : attention concentrée sur peu d'heures. "
                                        "Température haute : attention diffuse.")
    weights = M.attention_weights(reps, query, WINDOW, temperature)
    top = storm["time"].iloc[query - WINDOW + int(np.argmax(weights))]
    st.metric("Heure la plus regardée", f"{top:%d/%m %Hh}")
with chart:
    ui.plot(P.attention_plot(storm["time"], storm["dst"].to_numpy(), query, WINDOW, weights))
ui.intuition(
    "idéal pour les séquences (texte, séries temporelles), car chaque élément peut regarder tous les autres, "
    "même lointains, et choisir lesquels comptent."
)

with ui.aller_plus_loin():
    st.latex(r"\text{poids}_j = \frac{\exp(q \cdot k_j / T)}{\sum_{j'} \exp(q \cdot k_{j'} / T)}")
    st.markdown(
        "$q$ représente l'instant à prédire, $k_j$ chaque heure passée, $T$ la température. "
        "Dans un vrai Transformer, ces représentations sont **apprises**, et il y a plusieurs « têtes » "
        "d'attention en parallèle."
    )

# ------------------------------------------------------------
st.header("5.5 Récapitulatif")
st.table(pd.DataFrame(
    [
        ("Tableau de features (V, Bz, N, heure…)", "MLP", "Kp à partir du vent solaire"),
        ("Cartes, images", "CNN", "Cartes de TEC"),
        ("Séquences", "Transformer", "Séries temporelles d'indices"),
    ],
    columns=["Type de données", "Architecture", "Exemple"],
).set_index("Type de données"))
st.caption("Pour un tableau de features, les modèles à base d'arbres (random forest, gradient boosting) "
           "sont aussi d'excellents points de départ.")
st.markdown("Les modèles d'Augura Space **combinent CNN et Transformer** pour traiter des **séquences de cartes**.")

ui.a_retenir(
    [
        "Un neurone = somme pondérée + biais + activation ; un MLP = des couches de neurones.",
        "CNN : des filtres appris qui glissent sur une carte. Transformer : de l'attention sur une séquence.",
        "On choisit l'architecture selon la forme des données.",
    ]
)
