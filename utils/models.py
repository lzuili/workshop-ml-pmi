"""Modèles du workshop : régression du café, descente de gradient, modèles scikit-learn,
MLP PyTorch, convolution et attention illustrative.

Fonctions pures (sans Streamlit), partagées par l'app et le notebook.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from torch import nn

SEED = 42

# Nos réseaux sont minuscules : un seul thread est ici 5 à 20 fois plus rapide que plusieurs
torch.set_num_threads(1)

# ============================================================
# 1. Régression linéaire et descente de gradient (module 1)
# ============================================================


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mse_grid(x: np.ndarray, y: np.ndarray, a_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """MSE de la droite y = a*x + b pour chaque couple (a, b). Forme : (len(b), len(a))."""
    a = a_values[None, :, None]
    b = b_values[:, None, None]
    return np.mean((a * x[None, None, :] + b - y[None, None, :]) ** 2, axis=-1)


class LineGradientDescent:
    """Descente de gradient sur la droite y = a*x + b, pas à pas.

    normalize=True : on travaille sur la température normalisée x' = (x - moyenne) / écart-type,
    avec la droite y = a'*x' + b'. Les paramètres de travail sont alors (a', b').
    `to_original` reconvertit (a', b') en (a, b) pour prédire en °C.
    """

    MAX_LOSS = 1e12  # au-delà, on considère que la descente a divergé

    def __init__(self, x: np.ndarray, y: np.ndarray, normalize: bool, start=(0.0, 0.0)):
        self.normalize = normalize
        self.mean, self.std = (x.mean(), x.std()) if normalize else (0.0, 1.0)
        self.x = (x - self.mean) / self.std
        self.y = y
        self.params = [np.array(start, dtype=float)]
        self.losses = [self.loss(self.params[0])]

    def loss(self, params: np.ndarray) -> float:
        a, b = params
        return mse(self.y, a * self.x + b)

    def gradient(self, params: np.ndarray) -> np.ndarray:
        a, b = params
        error = a * self.x + b - self.y
        return np.array([2 * np.mean(error * self.x), 2 * np.mean(error)])

    @property
    def diverged(self) -> bool:
        return not np.isfinite(self.losses[-1]) or self.losses[-1] > self.MAX_LOSS

    def step(self, learning_rate: float, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            if self.diverged:
                return
            new = self.params[-1] - learning_rate * self.gradient(self.params[-1])
            self.params.append(new)
            self.losses.append(self.loss(new))

    def to_original(self, params: np.ndarray) -> tuple[float, float]:
        a, b = params
        return a / self.std, b - a * self.mean / self.std

    def trajectory(self) -> np.ndarray:
        return np.array(self.params)


# ============================================================
# 2. Sur-apprentissage (module 2)
# ============================================================


def fit_poly(x: np.ndarray, y: np.ndarray, degree: int) -> np.polynomial.Polynomial:
    """Polynôme de degré donné (Polynomial.fit reste stable numériquement jusqu'au degré 15)."""
    return np.polynomial.Polynomial.fit(x, y, degree)


def poly_errors(data: dict, degrees) -> pd.DataFrame:
    rows = []
    for degree in degrees:
        model = fit_poly(data["x_train"], data["y_train"], degree)
        rows.append(
            {
                "degre": degree,
                "train": mse(data["y_train"], model(data["x_train"])),
                "validation": mse(data["y_val"], model(data["x_val"])),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# 3. Modèles scikit-learn (module 4)
# ============================================================


def train_forest(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=60, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    return model.fit(X, y)


def train_logreg(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(X, y)


# ============================================================
# 4. Réseaux de neurones (module 5)
# ============================================================

ACTIVATIONS = {
    "aucune": lambda z: z,
    "ReLU": lambda z: np.maximum(z, 0),
    "sigmoïde": lambda z: 1 / (1 + np.exp(-z)),
}


def neuron(x: np.ndarray, weight: float, bias: float, activation: str) -> np.ndarray:
    """Un neurone : somme pondérée + biais, puis fonction d'activation."""
    return ACTIVATIONS[activation](weight * x + bias)


def set_seed(seed: int = SEED) -> None:
    """Graines fixées (reproductibilité). Rappelle aussi le réglage à 1 thread : il ne vaut que pour
    le thread qui l'appelle, et Streamlit exécute les pages dans un autre thread."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class MLP(nn.Module):
    """Réseau de neurones multicouche : [Linear -> ReLU] x couches cachées, puis Linear."""

    def __init__(self, n_inputs: int, hidden_sizes: list[int], n_outputs: int = 1):
        super().__init__()
        layers = []
        size_in = n_inputs
        for size in hidden_sizes:
            layers += [nn.Linear(size_in, size), nn.ReLU()]
            size_in = size
        layers.append(nn.Linear(size_in, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_1d(
    x: np.ndarray,
    y: np.ndarray,
    hidden_sizes: list[int],
    epochs: int = 300,
    learning_rate: float = 0.01,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[MLP, list[float], list[float], tuple[float, float, float, float]]:
    """Entraîne un petit MLP sur un signal 1D (tout le jeu en un seul batch).

    Renvoie le modèle, la loss train et la loss validation à chaque époque (dans l'unité² de y),
    et la normalisation (moyennes, écarts-types).
    """
    set_seed(seed)
    x_mean, x_std, y_mean, y_std = x.mean(), x.std(), y.mean(), y.std()
    X = torch.tensor((x - x_mean) / x_std, dtype=torch.float32)[:, None]
    Y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)[:, None]
    has_val = x_val is not None
    if has_val:
        X_val = torch.tensor((x_val - x_mean) / x_std, dtype=torch.float32)[:, None]
        Y_val = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32)[:, None]

    model = MLP(1, hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses, val_losses = [], []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item() * y_std**2)
        if has_val:
            with torch.no_grad():
                val_losses.append(loss_fn(model(X_val), Y_val).item() * y_std**2)
    return model, losses, val_losses, (x_mean, x_std, y_mean, y_std)


def predict_mlp_1d(model: MLP, x: np.ndarray, scaling: tuple[float, float, float, float]) -> np.ndarray:
    x_mean, x_std, y_mean, y_std = scaling
    with torch.no_grad():
        out = model(torch.tensor((x - x_mean) / x_std, dtype=torch.float32)[:, None])
    return out.numpy().ravel() * y_std + y_mean


def count_parameters(n_inputs: int, hidden_sizes: list[int], n_outputs: int = 1) -> int:
    sizes = [n_inputs, *hidden_sizes, n_outputs]
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


# ============================================================
# 5. Convolution (module 5.3)
# ============================================================

KERNELS = {
    "flou": np.full((3, 3), 1 / 9),
    "contours horizontaux": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float),
    "contours verticaux": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float),
}


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution 2D « valid » (sans bord), écrite simplement : le filtre glisse sur l'image."""
    kh, kw = kernel.shape
    out_h, out_w = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * image[i : i + out_h, j : j + out_w]
    return out


# ============================================================
# 6. Attention (module 5.4, illustration non entraînée)
# ============================================================


def attention_representations(dst: np.ndarray) -> np.ndarray:
    """Représentation construite à la main pour chaque heure : niveau de Dst et variation sur 3 h,
    tous deux standardisés."""
    level = (dst - dst.mean()) / dst.std()
    change = np.diff(dst, n=1, prepend=dst[0])
    change = pd.Series(change).rolling(3, min_periods=1).mean().to_numpy()
    change = (change - change.mean()) / change.std()
    return np.stack([level, change], axis=1)


def attention_weights(reps: np.ndarray, query_index: int, window: int, temperature: float) -> np.ndarray:
    """Poids d'attention de l'heure `query_index` sur les `window` heures précédentes.

    score = produit scalaire (similarité) / température, puis softmax (poids positifs, somme = 1).
    """
    keys = reps[query_index - window : query_index]
    scores = keys @ reps[query_index] / temperature
    scores -= scores.max()  # stabilité numérique
    weights = np.exp(scores)
    return weights / weights.sum()
