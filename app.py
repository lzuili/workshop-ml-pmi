"""Application du workshop : lancer avec `uv run streamlit run app.py`."""

import streamlit as st

st.set_page_config(page_title="Workshop Machine Learning", page_icon="🛰️", layout="wide")


def home() -> None:
    st.title("🛰️ Introduction au Machine Learning")
    st.markdown(
        "Workshop PMI IPSA x Augura Space.\n\n"
        "**Objectif** : comprendre comment un modèle apprend à partir d'exemples, "
        "et acquérir les bons réflexes avant d'entraîner vos propres modèles sur des données de météo spatiale."
    )

    st.subheader("Plan")
    st.markdown(
        """
| Module | Contenu | Durée |
|---|---|---|
| 1. Introduction | ML vs programmation classique, l'exemple du café, la descente de gradient | 12 min |
| 2. Notions clés | Vocabulaire, régression vs classification, sous et sur-apprentissage | 8 min |
| 3. Préparation des données | Valeurs manquantes, valeurs extrêmes, normalisation, variables cycliques | 10 min |
| 4. Évaluer un modèle | Train / validation / test, métriques de régression et de classification | 12 min |
| 5. Architectures | Neurone, MLP, CNN, Transformer | 10 min |
"""
    )
    st.caption("Utilisez le menu de gauche pour naviguer entre les modules.")

    st.subheader("Le notebook")
    st.markdown(
        "Les mêmes notions, avec le code visible : ouvrez `notebook/workshop_ml.ipynb` dans VS Code "
        "(kernel `.venv`), ou lancez :"
    )
    st.code("uv run jupyter lab notebook/workshop_ml.ipynb", language="bash")


pages = [
    st.Page(home, title="Accueil", icon="🏠", default=True),
    st.Page("modules/1_Introduction.py", title="1. Introduction", icon="☕"),
    st.Page("modules/2_Notions_cles.py", title="2. Notions clés", icon="📖"),
    st.Page("modules/3_Preparation_des_donnees.py", title="3. Préparation des données", icon="🧹"),
    st.Page("modules/4_Evaluation.py", title="4. Évaluer un modèle", icon="🎯"),
    st.Page("modules/5_Architectures.py", title="5. Architectures", icon="🧠"),
]
st.navigation(pages).run()
