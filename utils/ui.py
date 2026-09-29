"""Gabarit commun des pages Streamlit (importé uniquement par l'app, pas par le notebook)."""

from contextlib import contextmanager

import streamlit as st

from utils import data as D

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def page_header(title: str, en_bref: list[str]) -> None:
    st.title(title)
    st.markdown("**En bref**\n" + "\n".join(f"- {point}" for point in en_bref))


def intuition(text: str) -> None:
    st.info(f"💡 **Intuition** : {text}")


def a_retenir(points: list[str]) -> None:
    st.divider()
    st.success("**À retenir**\n" + "\n".join(f"- {point}" for point in points))


@contextmanager
def aller_plus_loin():
    with st.expander("Pour aller plus loin"):
        yield


def demo_columns():
    """Contrôles dans une colonne étroite à gauche, graphique à droite."""
    return st.columns([1, 3], gap="large")


def plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


@st.cache_data(show_spinner="Chargement des données OMNI…")
def omni_raw():
    return D.load_omni_raw()


@st.cache_data(show_spinner="Préparation des données OMNI…")
def omni_dataset():
    return D.load_dataset()
