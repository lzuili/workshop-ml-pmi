"""Fonctions de tracé Plotly communes à l'app et au notebook.

Polices et marqueurs dimensionnés pour un vidéoprojecteur (police de 16 px).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT_SIZE = 16
MARKER_SIZE = 12
# Dates en format français : jj/mm, avec l'heure quand on zoome sur quelques jours
DATE_FORMAT_STOPS = [
    dict(dtickrange=[None, 86_400_000], value="%d/%m<br>%Hh"),
    dict(dtickrange=[86_400_000, "M12"], value="%d/%m"),
    dict(dtickrange=["M12", None], value="%Y"),
]

COLORS = {
    "data": "#2a78d6",  # bleu : données observées / train
    "model": "#eb6834",  # orange : modèle / prédiction / validation
    "test": "#1baf7a",  # aqua : test
    "highlight": "#e34948",  # rouge : erreur, orage, point remarquable
    "violet": "#4a3aa7",
    "yellow": "#eda100",
    "gray": "#8a8984",
    "ink": "#0b0b0b",
    "muted": "#52514e",
}
SPLIT_COLORS = {"train": COLORS["data"], "val": COLORS["model"], "test": COLORS["test"]}
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}

# Rampe séquentielle bleue (clair = erreur faible... inversée plus bas pour « fond de vallée » sombre)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
VALLEY_SCALE = [[i / (len(BLUES) - 1), c] for i, c in enumerate(reversed(BLUES))]
DIVERGING = [[0, "#184f95"], [0.5, "#f0efec"], [1, "#e34948"]]


def base_layout(fig: go.Figure, title: str | None = None, height: int = 450, **kwargs) -> go.Figure:
    kwargs.setdefault("margin", dict(l=60, r=20, t=60 if title else 20, b=60))
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)) if title else None,
        font=dict(size=FONT_SIZE, color=COLORS["ink"]),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=FONT_SIZE),
        **kwargs,
    )
    return fig


# ============================================================
# Module 1 : café
# ============================================================


def coffee_plot(
    df: pd.DataFrame,
    line: tuple[float, float] | None = None,
    residuals: bool = False,
    unknown_day: int = 7,
    x_range=(5, 42),
) -> go.Figure:
    """Nuage de points du café, droite éventuelle, résidus, et jour inconnu mis en évidence."""
    known = df.dropna()
    unknown = df[df["ventes"].isna()]
    fig = go.Figure()

    if line is not None:
        a, b = line
        xs = np.array(x_range)
        if residuals:
            for x, y in zip(known["temperature"], known["ventes"]):
                fig.add_trace(
                    go.Scatter(
                        x=[x, x], y=[y, a * x + b], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color=COLORS["highlight"], width=2, dash="dot"),
                    )
                )
            fig.add_trace(  # entrée de légende unique pour les résidus
                go.Scatter(x=[None], y=[None], mode="lines", name="erreurs (résidus)",
                           line=dict(color=COLORS["highlight"], width=2, dash="dot"))
            )
        fig.add_trace(
            go.Scatter(x=xs, y=a * xs + b, mode="lines", name=f"y = {a:.2f}·x + {b:.1f}",
                       line=dict(color=COLORS["model"], width=3))
        )

    fig.add_trace(
        go.Scatter(
            x=known["temperature"], y=known["ventes"], mode="markers", name="jours connus",
            marker=dict(size=MARKER_SIZE + 2, color=COLORS["data"], line=dict(width=2, color="white")),
            customdata=known["jour"], hovertemplate="jour %{customdata}<br>%{x} °C : %{y} tasses<extra></extra>",
        )
    )
    for _, row in unknown.iterrows():
        x = row["temperature"]
        y = line[0] * x + line[1] if line is not None else None
        fig.add_vline(x=x, line=dict(color=COLORS["gray"], dash="dash", width=2))
        if y is not None:
            fig.add_trace(
                go.Scatter(x=[x], y=[y], mode="markers", name=f"jour {unknown_day} prédit",
                           marker=dict(size=MARKER_SIZE + 6, color=COLORS["model"], symbol="star",
                                       line=dict(width=1, color="white")))
            )
        fig.add_annotation(x=x, y=470, xanchor="left", yanchor="top", xshift=6, text=f"<b>?</b> jour {unknown_day}",
                           showarrow=False, font=dict(size=FONT_SIZE + 4, color=COLORS["highlight"]))

    fig.update_xaxes(title="température (°C)", range=list(x_range))
    fig.update_yaxes(title="ventes (tasses)", range=[150, 480])
    return base_layout(fig)


def valley_plot(
    a_values: np.ndarray,
    b_values: np.ndarray,
    mse_values: np.ndarray,
    minimum: tuple[float, float],
    current: tuple[float, float] | None = None,
    trajectory: np.ndarray | None = None,
    axis_titles=("a (pente)", "b (ordonnée à l'origine)"),
    equal_axes: bool = False,
    height: int = 480,
) -> go.Figure:
    """Carte de niveaux de l'erreur (échelle log) en fonction des deux paramètres."""
    fig = go.Figure(
        go.Contour(
            x=a_values, y=b_values, z=np.log10(mse_values), colorscale=VALLEY_SCALE, ncontours=25,
            contours=dict(coloring="heatmap", showlines=True), line=dict(width=0.5, color="white"),
            colorbar=dict(title="log₁₀(MSE)", thickness=15),
            customdata=mse_values, hovertemplate="a=%{x:.2f}<br>b=%{y:.1f}<br>MSE=%{customdata:.0f}<extra></extra>",
        )
    )
    if trajectory is not None and len(trajectory):
        fig.add_trace(
            go.Scatter(x=trajectory[:, 0], y=trajectory[:, 1], mode="lines+markers", name="descente",
                       line=dict(color=COLORS["model"], width=2), marker=dict(size=7, color=COLORS["model"]))
        )
        current = tuple(trajectory[-1])
    fig.add_trace(
        go.Scatter(x=[minimum[0]], y=[minimum[1]], mode="markers", name="minimum",
                   marker=dict(size=MARKER_SIZE + 8, symbol="star", color=COLORS["yellow"],
                               line=dict(width=1.5, color=COLORS["ink"])))
    )
    if current is not None:
        fig.add_trace(
            go.Scatter(x=[current[0]], y=[current[1]], mode="markers", name="position actuelle",
                       marker=dict(size=MARKER_SIZE + 4, color=COLORS["highlight"], line=dict(width=2, color="white")))
        )
    fig.update_xaxes(title=axis_titles[0], range=[a_values[0], a_values[-1]])
    fig.update_yaxes(title=axis_titles[1], range=[b_values[0], b_values[-1]])
    if equal_axes:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
        fig.update_xaxes(constrain="domain")
    return base_layout(fig, height=height)


def loss_curve(losses, title: str | None = None, log_y: bool = True, height: int = 320,
               name: str = "loss", val_losses=None) -> go.Figure:
    fig = go.Figure(
        go.Scatter(y=losses, mode="lines+markers", name=name, line=dict(color=COLORS["data"], width=3),
                   marker=dict(size=5))
    )
    if val_losses is not None:
        fig.add_trace(go.Scatter(y=val_losses, mode="lines", name="validation",
                                 line=dict(color=COLORS["model"], width=3)))
    fig.update_xaxes(title="itération")
    fig.update_yaxes(title="erreur (MSE)", type="log" if log_y else "linear", dtick=1 if log_y else None)
    return base_layout(fig, title=title, height=height, showlegend=val_losses is not None)


# ============================================================
# Module 2 : notions clés
# ============================================================


def regression_vs_classification(seed: int = 0) -> go.Figure:
    rng = np.random.default_rng(seed)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Régression : prédire un nombre",
                                                        "Classification : prédire une catégorie"))
    x = rng.uniform(0, 10, 30)
    y = 0.8 * x + 1 + rng.normal(0, 0.8, 30)
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=MARKER_SIZE, color=COLORS["data"]),
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=[0, 10], y=[1, 9], mode="lines", line=dict(color=COLORS["model"], width=3),
                             showlegend=False), 1, 1)
    for label, center, color, symbol in [("calme", (3, 3), COLORS["data"], "circle"),
                                         ("orage", (7, 7), COLORS["highlight"], "diamond")]:
        pts = rng.normal(center, 1.1, size=(25, 2))
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=label,
                                 marker=dict(size=MARKER_SIZE, color=color, symbol=symbol)), 1, 2)
    fig.add_trace(go.Scatter(x=[1, 9], y=[9, 1], mode="lines", name="frontière",
                             line=dict(color=COLORS["ink"], width=3, dash="dash")), 1, 2)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_annotations(font_size=FONT_SIZE + 1)
    fig = base_layout(fig, height=420)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="right", x=1))
    return fig


def poly_fit_plot(data: dict, model, degree: int, curve=None) -> go.Figure:
    xs = np.linspace(0, 1, 400)
    fig = go.Figure()
    if curve is not None:
        fig.add_trace(go.Scatter(x=xs, y=curve(xs), mode="lines", name="vraie tendance",
                                 line=dict(color=COLORS["gray"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=model(xs), mode="lines", name=f"polynôme de degré {degree}",
                             line=dict(color=COLORS["model"], width=3)))
    fig.add_trace(go.Scatter(x=data["x_train"], y=data["y_train"], mode="markers", name="train",
                             marker=dict(size=MARKER_SIZE, color=COLORS["data"])))
    fig.add_trace(go.Scatter(x=data["x_val"], y=data["y_val"], mode="markers", name="validation",
                             marker=dict(size=MARKER_SIZE, color="white", line=dict(width=3, color=COLORS["model"]))))
    fig.update_xaxes(title="x")
    fig.update_yaxes(title="y", range=[-2.2, 2.6])
    return base_layout(fig, height=420)


def poly_error_plot(errors: pd.DataFrame, degree: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=errors["degre"], y=errors["train"], mode="lines+markers", name="erreur train",
                             line=dict(color=COLORS["data"], width=3), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=errors["degre"], y=errors["validation"], mode="lines+markers",
                             name="erreur validation", line=dict(color=COLORS["model"], width=3),
                             marker=dict(size=9, symbol="circle-open", line=dict(width=3))))
    fig.add_vline(x=degree, line=dict(color=COLORS["ink"], dash="dash", width=2))
    fig.update_xaxes(title="degré du polynôme (complexité)", dtick=1)
    fig.update_yaxes(title="erreur (MSE, échelle log)", type="log", dtick=1)
    return base_layout(fig, height=420)


# ============================================================
# Module 3 : préparation des données
# ============================================================


def time_series(
    df: pd.DataFrame, col: str, y_title: str, highlight=None, highlight_name: str = "",
    height: int = 380, color: str = COLORS["data"], name: str | None = None,
) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["time"], y=df[col], mode="lines", name=name or col,
                               line=dict(color=color, width=2)))
    if highlight is not None and highlight.any():
        sub = df[highlight]
        fig.add_trace(go.Scatter(x=sub["time"], y=sub[col], mode="markers", name=highlight_name,
                                 marker=dict(size=MARKER_SIZE - 2, color=COLORS["highlight"])))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(tickformatstops=DATE_FORMAT_STOPS)
    return base_layout(fig, height=height, showlegend=highlight is not None)


def histograms(df: pd.DataFrame, cols: list[str], names: list[str], x_title: str) -> go.Figure:
    fig = go.Figure()
    for col, name, color in zip(cols, names, [COLORS["data"], COLORS["model"]]):
        fig.add_trace(go.Histogram(x=df[col], name=name, marker_color=color, opacity=0.65, nbinsx=80))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="nombre d'heures")
    return base_layout(fig, height=380)


def hours_on_line(h1: int, h2: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(24)), y=[0] * 24, mode="markers+text",
                             text=[f"{h}h" if h % 3 == 0 or h in (h1, h2) else "" for h in range(24)],
                             textposition="bottom center", marker=dict(size=8, color=COLORS["gray"]),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[h1, h2], y=[0, 0], mode="lines+markers", showlegend=False,
                             line=dict(color=COLORS["highlight"], width=4),
                             marker=dict(size=MARKER_SIZE + 8, color=[COLORS["data"], COLORS["model"]])))
    fig.add_annotation(x=(h1 + h2) / 2, y=0.6, text=f"distance = {abs(h1 - h2)}", showarrow=False,
                       font=dict(size=FONT_SIZE + 4, color=COLORS["highlight"]))
    fig.update_xaxes(visible=False, range=[-1, 24])
    fig.update_yaxes(visible=False, range=[-1.2, 1.2])
    return base_layout(fig, title="Valeur brute (droite de 0 à 23)", height=300)


def values_on_circle(v1: float, v2: float, period: float, tick_labels: dict, title: str) -> go.Figure:
    angle = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure(go.Scatter(x=np.cos(angle), y=np.sin(angle), mode="lines", showlegend=False,
                               line=dict(color=COLORS["gray"], width=2), hoverinfo="skip"))
    for value, label in tick_labels.items():
        t = 2 * np.pi * value / period
        fig.add_annotation(x=1.22 * np.cos(t), y=1.22 * np.sin(t), text=label, showarrow=False)
    pts = np.array([[np.cos(2 * np.pi * v / period), np.sin(2 * np.pi * v / period)] for v in (v1, v2)])
    dist = float(np.linalg.norm(pts[0] - pts[1]))
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", showlegend=False,
                             line=dict(color=COLORS["highlight"], width=4),
                             marker=dict(size=MARKER_SIZE + 8, color=[COLORS["data"], COLORS["model"]])))
    fig.add_annotation(x=0, y=0, text=f"distance = {dist:.2f}", showarrow=False,
                       font=dict(size=FONT_SIZE + 4, color=COLORS["highlight"]))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5])
    fig.update_yaxes(visible=False, range=[-1.5, 1.5], scaleanchor="x")
    return base_layout(fig, title=title, height=380)


# ============================================================
# Module 4 : évaluation
# ============================================================


def split_timeline(times: pd.Series, labels: pd.Series, height: int = 260, marker_size: int = 4) -> go.Figure:
    fig = go.Figure()
    for split in ["train", "val", "test"]:
        mask = labels == split
        fig.add_trace(go.Scatter(x=times[mask], y=[SPLIT_NAMES[split]] * int(mask.sum()), mode="markers",
                                 name=SPLIT_NAMES[split],
                                 marker=dict(size=marker_size, color=SPLIT_COLORS[split], symbol="line-ns-open",
                                             line=dict(width=2, color=SPLIT_COLORS[split]))))
    fig.update_yaxes(categoryorder="array", categoryarray=["test", "validation", "train"])
    return base_layout(fig, height=height)


def zoom_split(df: pd.DataFrame, labels: pd.Series, col: str, y_title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["time"], y=df[col], mode="lines", line=dict(color=COLORS["gray"], width=1.5),
                               showlegend=False, hoverinfo="skip"))
    for split in ["train", "val", "test"]:
        mask = labels == split
        fig.add_trace(go.Scatter(x=df["time"][mask], y=df[col][mask], mode="markers", name=SPLIT_NAMES[split],
                                 marker=dict(size=9, color=SPLIT_COLORS[split],
                                             symbol="diamond" if split == "test" else "circle",
                                             line=dict(width=1.5, color="white"))))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(tickformatstops=DATE_FORMAT_STOPS)
    return base_layout(fig, height=360)


def true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6],
                        subplot_titles=("Prédit en fonction du vrai", "Au cours du temps"))
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="prédiction parfaite",
                             line=dict(color=COLORS["gray"], dash="dash", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", showlegend=False,
                             marker=dict(size=8, color=COLORS["model"], opacity=0.8)), 1, 1)
    t = np.arange(len(y_true))
    fig.add_trace(go.Scatter(x=t, y=y_true, mode="lines", name="vrai", line=dict(color=COLORS["data"], width=3)), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=y_pred, mode="lines", name="prédit", line=dict(color=COLORS["model"], width=2)), 1, 2)
    fig.update_xaxes(title_text="vrai", row=1, col=1)
    fig.update_yaxes(title_text="prédit", row=1, col=1)
    fig.update_xaxes(title_text="heure", row=1, col=2)
    fig.update_annotations(font_size=FONT_SIZE + 1)
    fig = base_layout(fig, height=440)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=1))
    return fig


def confusion_matrix_plot(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    """Matrice de confusion avec libellés en clair."""
    z = [[tp, fn], [fp, tn]]
    text = [[f"<b>{tp}</b><br>orage<br>annoncé<br>et réel", f"<b>{fn}</b><br>orage<br>manqué"],
            [f"<b>{fp}</b><br>fausse<br>alerte", f"<b>{tn}</b><br>calme<br>annoncé<br>et réel"]]
    colors = [[COLORS["test"], COLORS["highlight"]], [COLORS["yellow"], "#cde2fb"]]
    fig = go.Figure()
    for i in range(2):
        for j in range(2):
            fig.add_shape(type="rect", x0=j, x1=j + 1, y0=1 - i, y1=2 - i, fillcolor=colors[i][j],
                          opacity=0.85, line=dict(color="white", width=4))
            fig.add_annotation(x=j + 0.5, y=1.5 - i, text=text[i][j], showarrow=False,
                               font=dict(size=FONT_SIZE + 1, color=COLORS["ink"]))
    fig.update_xaxes(range=[0, 2], tickvals=[0.5, 1.5], ticktext=["réel :<br>orage", "réel :<br>calme"],
                     side="top", showgrid=False, zeroline=False, tickfont=dict(size=FONT_SIZE))
    fig.update_yaxes(range=[0, 2], tickvals=[1.5, 0.5], ticktext=["prédit :<br>orage", "prédit :<br>calme"],
                     showgrid=False, zeroline=False, tickfont=dict(size=FONT_SIZE))
    return base_layout(fig, height=420, margin=dict(l=80, r=10, t=60, b=10))


def storm_probability_plot(df: pd.DataFrame, proba: np.ndarray, threshold: float | None,
                           target: str = "storm_next") -> go.Figure:
    """Probabilité d'orage prédite au cours du temps, heures d'orage réel, et seuil de décision."""
    fig = go.Figure()
    storm = df[target].to_numpy() == 1
    fig.add_trace(go.Bar(x=df["time"][storm], y=[1.0] * int(storm.sum()), name="orage réel",
                         marker_color=COLORS["highlight"], opacity=0.25, width=3_600_000))
    fig.add_trace(go.Scatter(x=df["time"], y=proba, mode="lines", name="probabilité prédite",
                             line=dict(color=COLORS["data"], width=3)))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=COLORS["ink"], dash="dash", width=2),
                      annotation_text=f"seuil {threshold:.2f}", annotation_position="top left")
    fig.update_yaxes(title="probabilité d'orage", range=[0, 1.02])
    fig.update_xaxes(tickformatstops=DATE_FORMAT_STOPS)
    fig = base_layout(fig, title="Mai 2024 (test)", height=420)
    fig.update_layout(bargap=0, legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    return fig


# ============================================================
# Module 5 : architectures
# ============================================================


def neuron_plot(x: np.ndarray, y: np.ndarray, activation: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=COLORS["model"], width=4),
                               name="sortie du neurone"))
    fig.add_hline(y=0, line=dict(color=COLORS["gray"], width=1))
    fig.add_vline(x=0, line=dict(color=COLORS["gray"], width=1))
    fig.update_xaxes(title="entrée x")
    fig.update_yaxes(title=f"sortie (activation : {activation})", range=[-4, 4])
    return base_layout(fig, height=400)


def network_diagram(layer_sizes: list[int], max_drawn: int = 10) -> go.Figure:
    """Schéma d'un MLP : un nœud par neurone (au plus max_drawn par couche), liaisons entre couches."""
    fig = go.Figure()
    positions = []
    for i, size in enumerate(layer_sizes):
        drawn = min(size, max_drawn)
        ys = np.linspace(-(drawn - 1) / 2, (drawn - 1) / 2, drawn) if drawn > 1 else np.array([0.0])
        positions.append((i, ys, size > drawn))
    edge_x, edge_y = [], []
    for (x0, ys0, _), (x1, ys1, _) in zip(positions[:-1], positions[1:]):
        for y0 in ys0:
            for y1 in ys1:
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#c9c8c3", width=1),
                             hoverinfo="skip", showlegend=False))
    n_layers = len(layer_sizes)
    half = (max(len(ys) for _, ys, _ in positions) - 1) / 2
    for i, ys, truncated in positions:
        color = COLORS["data"] if i == 0 else COLORS["model"] if i == n_layers - 1 else COLORS["violet"]
        fig.add_trace(go.Scatter(x=[i] * len(ys), y=ys, mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=22, color=color, line=dict(width=2, color="white"))))
        name = "entrée" if i == 0 else "sortie" if i == n_layers - 1 else f"couche cachée {i}"
        label = f"{name}<br>{layer_sizes[i]} neurone{'s' if layer_sizes[i] > 1 else ''}"
        if truncated:
            label += f" ({max_drawn} dessinés)"
        fig.add_annotation(x=i, y=-half - 0.7, text=label, showarrow=False, yanchor="top",
                           font=dict(size=FONT_SIZE - 2))
    fig.update_xaxes(visible=False, range=[-0.5, n_layers - 0.5])
    fig.update_yaxes(visible=False, range=[-half - 2.2, half + 0.7])
    return base_layout(fig, height=120 + 32 * (2 * half + 3))


def mlp_fit_plot(x, y, xs, ys_pred, true_curve=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="données", marker=dict(size=7, color=COLORS["data"],
                                                                                  opacity=0.6)))
    if true_curve is not None:
        fig.add_trace(go.Scatter(x=xs, y=true_curve, mode="lines", name="vraie tendance",
                                 line=dict(color=COLORS["gray"], dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=ys_pred, mode="lines", name="MLP", line=dict(color=COLORS["model"], width=4)))
    fig.update_xaxes(title="heure locale")
    fig.update_yaxes(title="valeur (unité arbitraire)")
    return base_layout(fig, height=400)


def image_plot(img: np.ndarray, title: str, window=None, pixel=None, colorscale="Viridis",
               zmid=None, height: int = 380, extent=((-90, 90), (-180, 180))) -> go.Figure:
    """Affiche une image 2D. window = (ligne, colonne, taille) : cadre du filtre ; pixel = (ligne, colonne)."""
    fig = go.Figure(go.Heatmap(z=img, colorscale=colorscale, zmid=zmid, colorbar=dict(thickness=12),
                               hovertemplate="ligne %{y}, colonne %{x}<br>valeur %{z:.1f}<extra></extra>"))
    if window is not None:
        r, c, k = window
        fig.add_shape(type="rect", x0=c - 0.5, x1=c + k - 0.5, y0=r - 0.5, y1=r + k - 0.5,
                      line=dict(color=COLORS["highlight"], width=4))
    if pixel is not None:
        r, c = pixel
        fig.add_shape(type="rect", x0=c - 1.5, x1=c + 1.5, y0=r - 1.5, y1=r + 1.5,
                      line=dict(color=COLORS["highlight"], width=4))
    fig.update_xaxes(showticklabels=False, title="longitude →", constrain="domain", showgrid=False)
    fig.update_yaxes(showticklabels=False, title="latitude →", scaleanchor="x", constrain="domain", showgrid=False)
    return base_layout(fig, title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))


def attention_plot(times: pd.Series, dst: np.ndarray, query_index: int, window: int, weights: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=times, y=dst, mode="lines", name="Dst", line=dict(color=COLORS["data"], width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=[times.iloc[query_index]], y=[dst[query_index]], mode="markers",
                             name="instant à prédire", marker=dict(size=MARKER_SIZE + 6, color=COLORS["model"],
                                                                    symbol="star", line=dict(width=1, color="white"))),
                  1, 1)
    past = times.iloc[query_index - window : query_index]
    fig.add_trace(go.Bar(x=past, y=weights, name="poids d'attention", marker_color=COLORS["model"]), 2, 1)
    fig.add_vrect(x0=past.iloc[0], x1=times.iloc[query_index], fillcolor=COLORS["model"], opacity=0.08,
                  line_width=0, row=1, col=1)
    fig.update_yaxes(title_text="Dst (nT)", row=1, col=1)
    fig.update_yaxes(title_text="attention", row=2, col=1, rangemode="tozero")
    fig.update_xaxes(tickformatstops=DATE_FORMAT_STOPS)
    return base_layout(fig, height=520)
