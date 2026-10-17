"""
=============================================================================
verifier_resultats.py -- Comparaison outputs/ vs resultats_reference/
=============================================================================
A lancer apres `python reproduire.py`. Compare chaque fichier de
resultats_reference/ a son homologue regenere dans outputs/ :

  - CSV  : memes colonnes, memes dimensions, valeurs numeriques comparees
           (identiques, ou ecart relatif <= TOL_REL)
  - JSON : comparaison recursive, en ignorant les horodatages
  - GAML / .sir : comparaison texte, en ignorant les lignes de date
  - PNG  : presence seulement (le rendu binaire varie selon la plateforme)

Code de sortie 0 si tout concorde, 1 sinon.
=============================================================================
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

REF = "resultats_reference"
OUT = "outputs"
TOL_REL = 1e-6          # ecart relatif tolere (arrondis flottants inter-plateformes)
CLES_IGNOREES = {"timestamp", "calibration_date"}


def comparer_csv(ref, out):
    a, b = pd.read_csv(ref), pd.read_csv(out)
    if list(a.columns) != list(b.columns):
        return False, "colonnes differentes"
    if a.shape != b.shape:
        return False, f"dimensions {b.shape} au lieu de {a.shape}"
    if a.equals(b):
        return True, "identique"
    num = a.select_dtypes("number").columns
    autres = [c for c in a.columns if c not in num]
    if not a[autres].equals(b[autres]):
        return False, "colonnes non numeriques differentes"
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False, "valeurs manquantes differentes"
    m = ~np.isnan(x)
    ecart = np.abs(x[m] - y[m]) / np.maximum(np.abs(x[m]), 1e-12)
    ok = np.allclose(x[m], y[m], rtol=TOL_REL, atol=1e-12)
    return ok, f"ecart relatif max {ecart.max():.2e}"


def egal_json(a, b, chemin=""):
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        for k in set(a) | set(b):
            if k in CLES_IGNOREES:
                continue
            if k not in a or k not in b:
                diffs.append(f"{chemin}/{k} absent")
            else:
                diffs += egal_json(a[k], b[k], f"{chemin}/{k}")
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{chemin} : longueur {len(b)} au lieu de {len(a)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in egal_json(x, y, f"{chemin}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if math.isclose(a, b, rel_tol=TOL_REL, abs_tol=1e-12):
            return []
        return [f"{chemin} : {b} au lieu de {a}"]
    return [] if a == b else [f"{chemin} : {b!r} au lieu de {a!r}"]


def comparer_json(ref, out):
    with open(ref, encoding="utf-8") as f:
        a = json.load(f)
    with open(out, encoding="utf-8") as f:
        b = json.load(f)
    diffs = egal_json(a, b)
    return (not diffs), ("identique (hors horodatage)" if not diffs
                         else f"{len(diffs)} ecart(s), ex. {diffs[0]}")


def comparer_texte(ref, out):
    def lignes(p):
        with open(p, encoding="utf-8") as f:
            return [l for l in f if "Date" not in l and "calibration_date" not in l]
    ok = lignes(ref) == lignes(out)
    return ok, "identique (hors date)" if ok else "contenu different"


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(OUT):
        sys.exit("outputs/ introuvable : lancez d'abord `python reproduire.py`.")

    n_ok, echecs = 0, []
    for nom in sorted(os.listdir(REF)):
        ref, out = os.path.join(REF, nom), os.path.join(OUT, nom)
        if not os.path.exists(out):
            ok, msg = False, "absent de outputs/"
        elif nom.endswith(".csv"):
            ok, msg = comparer_csv(ref, out)
        elif nom.endswith(".json"):
            ok, msg = comparer_json(ref, out)
        elif nom.endswith((".gaml", ".sir")):
            ok, msg = comparer_texte(ref, out)
        else:
            ok, msg = True, "present"
        print(f"  [{'OK ' if ok else 'ECART'}] {nom:55s} {msg}")
        n_ok += ok
        if not ok:
            echecs.append(nom)

    total = n_ok + len(echecs)
    print(f"\n{n_ok}/{total} fichiers conformes aux resultats de reference.")
    if echecs:
        print("Fichiers en ecart :", ", ".join(echecs))
        sys.exit(1)


if __name__ == "__main__":
    main()
