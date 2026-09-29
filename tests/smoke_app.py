"""Test de fumée de l'app : chaque page s'affiche sans erreur et chaque démo réagit en moins d'1 s.

Lancer : uv run python tests/smoke_app.py [numéro de page ...]
"""

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = {
    1: "modules/1_Introduction.py",
    2: "modules/2_Notions_cles.py",
    3: "modules/3_Preparation_des_donnees.py",
    4: "modules/4_Evaluation.py",
    5: "modules/5_Architectures.py",
}
MAX_SECONDS = 1.0
failures = []


def check(at: AppTest, label: str, start: float, max_seconds: float = MAX_SECONDS) -> None:
    elapsed = time.perf_counter() - start
    status = "OK"
    if at.exception:
        status = "EXCEPTION"
        failures.append(f"{label} : {at.exception[0].message}")
    elif elapsed > max_seconds:
        status = "LENT"
        failures.append(f"{label} : {elapsed:.2f} s")
    print(f"  [{status}] {label} ({elapsed:.2f} s)")


def run(at: AppTest, label: str, max_seconds: float = MAX_SECONDS) -> AppTest:
    start = time.perf_counter()
    at.run()
    check(at, label, start, max_seconds)
    return at


def metric(at: AppTest, label: str) -> str:
    exact = [m.value for m in at.metric if m.label == label]
    return exact[0] if exact else next(m.value for m in at.metric if m.label.startswith(label))


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def slider(at: AppTest, label: str):
    return next(s for s in at.slider if s.label.startswith(label))


def select_slider(at: AppTest, label: str):
    return next(s for s in at.select_slider if s.label.startswith(label))


def open_page(number: int) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    at.switch_page(PAGES[number])
    return run(at, f"page {number} : premier affichage", max_seconds=120)


def expect(condition: bool, message: str) -> None:
    print(f"  [{'OK' if condition else 'ÉCHEC'}] {message}")
    if not condition:
        failures.append(message)


# ------------------------------------------------------------


def test_home():
    print("Accueil")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    run(at, "accueil", max_seconds=60)


def test_page_1():
    print("Page 1")
    at = open_page(1)
    run(at, "réaffichage (cache chaud)")

    slider(at, "a (pente)").set_value(10.0)
    run(at, "slider a")
    mse_manual = float(metric(at, "Erreur (MSE)").replace(" ", ""))
    button(at, "Montrer la meilleure droite").click()
    run(at, "bouton meilleure droite")
    mse_best = float(metric(at, "Erreur (MSE)").replace(" ", ""))
    expect(mse_best < mse_manual, f"meilleure droite : MSE {mse_best} < {mse_manual}")
    expect(metric(at, "Jour 7 prédit") == "313 tasses", f"jour 7 prédit : {metric(at, 'Jour 7 prédit')}")

    # Descente de gradient normalisée, learning rate 0.1 : converge
    button(at, "N pas").click()
    run(at, "N pas (normalisé, lr 0.1)")
    button(at, "N pas").click()
    run(at, "N pas (bis)")
    loss = float(metric(at, "Erreur actuelle").replace(" ", ""))
    expect(abs(loss - mse_best) < 1, f"normalisé lr 0.1, 40 pas : MSE {loss} ≈ minimum {mse_best}")

    # Sans normalisation, lr 0.1 : diverge
    next(c for c in at.checkbox if c.label.startswith("Normaliser")).uncheck()
    run(at, "décocher normalisation")
    button(at, "1 pas").click()
    run(at, "1 pas brut")
    button(at, "N pas").click()
    run(at, "N pas brut (lr 0.1)")
    expect(any("divergé" in e.value for e in at.error), "brut, lr 0.1 : divergence affichée")

    # Sans normalisation, lr 0.001 : lent
    button(at, "Réinitialiser").click()
    select_slider(at, "Learning rate").set_value(0.001)
    slider(at, "Nombre de pas").set_value(200)
    run(at, "réinitialiser + lr 0.001")
    for _ in range(3):
        button(at, "N pas").click()
        run(at, "N pas brut (200 pas)")
    loss = float(metric(at, "Erreur actuelle").replace(" ", ""))
    expect(loss > 5 * mse_best, f"brut lr 0.001, 600 pas : MSE {loss} encore loin du minimum {mse_best}")


def test_page_2():
    print("Page 2")
    at = open_page(2)
    expect(any("Sous-apprentissage" in w.value for w in at.warning), "degré 1 : sous-apprentissage")
    val = {}
    for degree in [5, 7, 12, 15]:
        slider(at, "Degré du polynôme").set_value(degree)
        run(at, f"degré {degree}")
        val[degree] = float(metric(at, "Erreur validation"))
    expect(any("Sur-apprentissage" in e.value for e in at.error), "degré 15 : sur-apprentissage")
    slider(at, "Degré du polynôme").set_value(1)
    run(at, "degré 1")
    val[1] = float(metric(at, "Erreur validation"))
    expect(val[5] < val[1] and val[5] < val[12] < val[15], f"validation en U : {val}")


def radio(at: AppTest, label: str):
    return next(r for r in at.radio if r.label == label)


def test_page_3():
    print("Page 3")
    at = open_page(3)
    run(at, "réaffichage (cache chaud)")
    expect(metric(at, "Heures manquantes") == "83 / 672", f"trous février 2023 : {metric(at, 'Heures manquantes')}")
    radio(at, "Affichage").set_value("nettoyé")
    run(at, "bascule nettoyé")
    radio(at, "Variable").set_value("n")
    run(at, "variable N")

    counts = {}
    for threshold in [1.0, 3.0, 5.0]:
        slider(at, "Seuil de z-score").set_value(threshold)
        run(at, f"z-score {threshold}")
        counts[threshold] = int(metric(at, "Points détectés"))
    expect(counts[1.0] > counts[3.0] > 0, f"points détectés selon le seuil : {counts}")
    expect(any("10/05" in c.value for c in at.caption), "z-score 5 : points détectés le 10/05 (orage)")

    for method in ["standardisation", "min-max", "brut"]:
        radio(at, "Méthode").set_value(method)
        run(at, f"normalisation {method}")

    slider(at, "Première heure").set_value(22)
    run(at, "heure 22")
    slider(at, "Premier jour").set_value(360)
    run(at, "jour 360")


def percent(value: str) -> float:
    return float(value.rstrip("%")) / 100


def test_page_4():
    print("Page 4")
    at = open_page(4)
    run(at, "réaffichage (cache chaud)")
    r2_temporal = float(metric(at, "R² sur le test"))
    radio(at, "Découpage").set_value("aléatoire")
    run(at, "split aléatoire")
    r2_random = float(metric(at, "R² sur le test"))
    expect(r2_random > r2_temporal + 0.05, f"split aléatoire trop optimiste : R² {r2_random} vs {r2_temporal}")

    slider(at, "Bruit du modèle").set_value(0.4)
    slider(at, "Une grosse erreur").set_value(0.0)
    run(at, "métriques régression")
    mae0, rmse0 = float(metric(at, "MAE")), float(metric(at, "RMSE"))
    slider(at, "Une grosse erreur").set_value(10.0)
    run(at, "grosse erreur isolée")
    mae1, rmse1 = float(metric(at, "MAE")), float(metric(at, "RMSE"))
    expect(rmse1 - rmse0 > 2 * (mae1 - mae0), f"la RMSE réagit plus que la MAE : MAE {mae0}→{mae1}, RMSE {rmse0}→{rmse1}")
    slider(at, "Bruit du modèle").set_value(1.5)
    run(at, "bruit fort")

    results = {}
    for threshold in [0.1, 0.5, 0.9]:
        slider(at, "Seuil de décision").set_value(threshold)
        run(at, f"seuil {threshold}")
        results[threshold] = (percent(metric(at, "Précision")), percent(metric(at, "Rappel")))
    expect(results[0.1][1] > results[0.5][1] > results[0.9][1], f"le rappel baisse quand le seuil monte : {results}")
    expect(results[0.1][0] < results[0.9][0], f"la précision monte avec le seuil : {results}")

    next(t for t in at.toggle if t.label.startswith("Toujours")).set_value(True)
    run(at, "toujours calme")
    expect(percent(metric(at, "Accuracy")) > 0.96 and metric(at, "Rappel") == "0.0%",
           f"toujours calme : accuracy {metric(at, 'Accuracy')}, rappel {metric(at, 'Rappel')}")


def final_losses(at: AppTest) -> tuple[float, float]:
    caption = next(c.value for c in at.caption if c.value.startswith("Loss finale"))
    train = float(caption.split("train ")[1].split(",")[0])
    val = float(caption.split("validation ")[1].split(" ")[0])
    return train, val


def test_page_5():
    print("Page 5")
    at = open_page(5)
    for activation in ["ReLU", "sigmoïde", "aucune"]:
        radio(at, "Activation").set_value(activation)
        run(at, f"activation {activation}")
    slider(at, "Poids").set_value(2.5)
    run(at, "poids du neurone")

    # 1 couche de 1 neurone : sous-apprentissage (le premier entraînement inclut l'initialisation de torch)
    slider(at, "Neurones par couche").set_value(1)
    run(at, "architecture 1 neurone")
    button(at, "Entraîner").click()
    run(at, "entraîner 1 neurone")
    small = final_losses(at)
    slider(at, "Couches cachées").set_value(3)
    slider(at, "Neurones par couche").set_value(32)
    run(at, "architecture 3 x 32 (schéma seul)")
    expect(not any(c.value.startswith("Loss finale") for c in at.caption), "changer d'architecture efface le résultat")
    button(at, "Entraîner").click()
    run(at, "entraîner 3 x 32")
    big = final_losses(at)
    expect(big[0] < small[0] and big[1] < small[1], f"le gros réseau apprend mieux : {small} → {big}")
    expect(big[0] < big[1], f"écart train < validation pour 3 x 32 : {big}")
    button(at, "Entraîner").click()
    run(at, "ré-entraîner 3 x 32 (cache)")

    values = set()
    for kernel in ["contours horizontaux", "contours verticaux", "flou"]:
        radio(at, "Filtre").set_value(kernel)
        run(at, f"filtre {kernel}")
        values.add(metric(at, "Pixel de sortie"))
    slider(at, "Position horizontale").set_value(100)
    run(at, "déplacer le filtre")
    expect(len(values) == 3, f"chaque filtre donne une sortie différente : {values}")

    slider(at, "Instant à prédire").set_value(120)
    run(at, "instant à prédire")
    select_slider(at, "Concentration").set_value(0.1)
    run(at, "température 0.1")
    select_slider(at, "Concentration").set_value(5.0)
    run(at, "température 5")


if __name__ == "__main__":
    selected = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4, 5]
    for number in selected:
        test = globals().get("test_home" if number == 0 else f"test_page_{number}")
        if test:
            test()
    print()
    print("ÉCHECS :" if failures else "Tout est OK.")
    for failure in failures:
        print(" -", failure)
    sys.exit(1 if failures else 0)
