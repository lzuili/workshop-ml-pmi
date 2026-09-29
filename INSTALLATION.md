# Installation : workshop Machine Learning

Objectif : lancer l'application du cours et le notebook sur votre ordinateur.
Pas besoin d'installer Python vous-même : **uv** s'en charge.

---

## Étape 1 : installer VS Code

1. Télécharger et installer VS Code : https://code.visualstudio.com
2. Ouvrir VS Code, aller dans l'onglet **Extensions** (`Ctrl+Shift+X`, ou `Cmd+Shift+X` sur Mac) et installer :
   - **Python** (Microsoft)
   - **Jupyter** (Microsoft)

## Étape 2 : installer uv

Ouvrir un terminal (dans VS Code : menu **Terminal → New Terminal**).

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

⚠️ **Fermer puis rouvrir le terminal** (voire VS Code), puis vérifier :
```bash
uv --version
```

## Étape 3 : récupérer le projet

- **Avec git** : `git clone https://github.com/lzuili/workshop-ml-pmi.git`, puis dans VS Code : **File → Open Folder…** → choisir le dossier `workshop-ml-pmi`.
- **Sans git** : dézipper `workshop_ml.zip` là où vous voulez, puis dans VS Code : **File → Open Folder…** → choisir le dossier obtenu.

## Étape 4 : installer l'environnement

Dans le terminal de VS Code (qui est déjà dans le dossier du projet) :
```bash
uv sync
```
Cette commande installe la bonne version de Python et toutes les bibliothèques dans un dossier `.venv`. Elle peut prendre quelques minutes la première fois (PyTorch est volumineux).

Vérifier que tout est en place :
```bash
uv run python -c "import torch, streamlit; print('OK')"
```

## Étape 5 : lancer l'application du cours

```bash
uv run streamlit run app.py
```
L'application s'ouvre dans le navigateur (sinon : http://localhost:8501). Elle fonctionne ensuite sans connexion internet.
Pour l'arrêter : `Ctrl+C` dans le terminal.

## Étape 6 : ouvrir le notebook

1. Dans VS Code, ouvrir `notebook/workshop_ml.ipynb`.
2. En haut à droite, cliquer sur **Select Kernel** → **Python Environments** → choisir **`.venv`**.
3. Exécuter la première cellule (`Shift+Enter`).

(Alternative sans VS Code : `uv run jupyter lab`)

---

## Récapitulatif

| Quoi | Commande | Quand |
|---|---|---|
| Installer l'environnement | `uv sync` | une seule fois (et après une mise à jour du projet) |
| Lancer l'app | `uv run streamlit run app.py` | à chaque fois |
| Lancer Jupyter hors VS Code | `uv run jupyter lab` | au besoin |
| Lancer un script | `uv run python mon_script.py` | au besoin |
| Ajouter une bibliothèque | `uv add nom_du_paquet` | au besoin |

💡 Toujours préfixer par `uv run` : c'est ce qui garantit d'utiliser l'environnement du projet. Pas besoin d'« activer » quoi que ce soit.

---

## En cas de problème

| Symptôme | Solution |
|---|---|
| `uv` n'est pas reconnu | Fermer et rouvrir le terminal (ou redémarrer VS Code). |
| Windows : « l'exécution de scripts est désactivée » | Utiliser exactement la commande de l'étape 2 (elle contient `-ExecutionPolicy ByPass`). |
| Le kernel `.venv` n'apparaît pas dans VS Code | Vérifier que `uv sync` a bien tourné, puis `Ctrl+Shift+P` → **Python: Select Interpreter** → choisir `.venv`, et relancer VS Code. |
| `Port 8501 is already in use` | Une autre app tourne déjà : la fermer, ou `uv run streamlit run app.py --server.port 8502`. |
| `uv sync` très long | Normal la première fois (téléchargement de PyTorch). Vérifier la connexion wifi. |
| Mac Intel : `torch` en version 2.2.2 | Normal : c'est la dernière version publiée pour les Mac Intel, le projet l'installe automatiquement. En cas d'erreur, prévenir Léa. |
| La page « 4. Évaluer un modèle » met une dizaine de secondes à s'afficher la première fois | Normal : elle entraîne deux modèles au premier affichage, puis tout est instantané. |
