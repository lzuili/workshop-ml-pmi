# Workshop : introduction au Machine Learning

Support du workshop d'1h du projet PMI IPSA x Augura Space (météo spatiale et GNSS).

- **Une application Streamlit** qui sert de support de cours, avec une démo interactive par notion.
- **Un notebook Jupyter** qui reprend les mêmes notions avec le code visible, dont un MLP PyTorch complet.

Les deux tournent en local, sans connexion internet : les données sont incluses dans le dépôt.

## Installation

Tout est détaillé pas à pas dans [INSTALLATION.md](INSTALLATION.md) (VS Code + uv). En résumé :

```bash
uv sync
```

## Lancer l'application

```bash
uv run streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501. Cinq modules, pour environ 50 minutes en live :

| Module | Contenu |
|---|---|
| 1. Introduction | ML vs programmation classique, l'exemple du café, le paysage de l'erreur, la descente de gradient |
| 2. Notions clés | Vocabulaire, régression vs classification, sous et sur-apprentissage |
| 3. Préparation des données | Valeurs manquantes, valeur aberrante ou événement, normalisation, encodage cos / sin |
| 4. Évaluer un modèle | Split temporel vs aléatoire, MAE / RMSE / R², accuracy / précision / rappel |
| 5. Architectures | Neurone, MLP, CNN, Transformer |

## Lancer le notebook

Dans VS Code : ouvrir `notebook/workshop_ml.ipynb` et choisir le kernel `.venv`. Ou bien :

```bash
uv run jupyter lab notebook/workshop_ml.ipynb
```

Le notebook s'exécute de bout en bout en moins d'une minute sur un portable (CPU). Pour le vérifier sans l'ouvrir :

```bash
uv run jupyter nbconvert --to notebook --execute notebook/workshop_ml.ipynb
```

## Les données

`data/omni_2015_2024_1h.csv` : données OMNI horaires de la NASA (jeu `OMNI2_H0_MRG1HR` sur CDAWeb), 2015 à 2024.
Colonnes : Kp × 10, Dst, F10.7, vitesse (V), Bz et densité (N) du vent solaire.

Le fichier est **brut** : il garde les valeurs de remplissage OMNI (9999, 999.9…), qui servent au module 3.
Le nettoyage est fait dans `utils/data.py`.

Pour régénérer le fichier (avec internet) :

```bash
uv run --group data python data/fetch_omni.py
```

## Organisation du projet

```
app.py                  page d'accueil et navigation
modules/                une page Streamlit par module (1 à 5)
utils/                  code partagé par l'app et le notebook
  data.py               jeu café, données synthétiques, chargement et préparation OMNI
  models.py             descente de gradient, modèles scikit-learn, MLP PyTorch, convolution, attention
  plots.py              graphiques Plotly communs
  ui.py                 gabarit des pages (uniquement pour l'app)
data/                   script de récupération et CSV OMNI
assets/                 illustrations extraites de la présentation d'origine
notebook/               le notebook
tests/smoke_app.py      test de l'app : chaque page s'affiche et chaque démo réagit en moins d'1 s
```

Les pages sont dans `modules/` et non `pages/` : un dossier `pages/` active l'ancienne navigation de Streamlit,
qui ignore les titres des pages et la mise en page large au premier affichage.

## Tester l'application

```bash
uv run python tests/smoke_app.py        # toutes les pages
uv run python tests/smoke_app.py 1 4    # seulement les modules 1 et 4
```

## Notes techniques

- PyTorch en version CPU : sous Linux, un index dédié évite de télécharger CUDA (plusieurs Go).
- `torch < 2.10` : les versions suivantes exigent macOS 14 sur les Mac Apple Silicon.
- Mac Intel : PyTorch ne publie plus de versions après la 2.2.2, qui impose `numpy < 2` ;
  `argon2-cffi-bindings` (dépendance de Jupyter) est limité à une version qui fournit un paquet précompilé.
- Les seeds sont fixées partout : l'app et le notebook donnent les mêmes résultats d'une exécution à l'autre.
