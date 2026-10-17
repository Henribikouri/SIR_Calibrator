"""
=============================================================================
reproduire.py -- Reproduction des resultats du memoire (7 entites)
=============================================================================
Relance, pour chacune des 7 entites etudiees, les deux points d'entree du
projet avec les arguments exacts ayant produit les resultats de reference :

  1. pipeline.py        (M1 -> M5 : figures M1/M2/M3/M5, rapports JSON, GAML)
  2. M6_comparaison.py  (CSV dataset_compare / theta_trajectoire /
                         dynamique_temporelle + FIG1/FIG2/FIG3)

Toutes les sorties sont ecrites dans outputs/. On les compare ensuite aux
resultats de reference avec :  python verifier_resultats.py

Usage :
  python reproduire.py                      # les 7 entites (~10 min)
  python reproduire.py --only "New York"    # une seule entite
  python reproduire.py --skip-pipeline      # M6 seul (CSV + FIG1/2/3)

L'ordre se termine par la Sierra Leone : les fichiers sans suffixe d'entite
(calibration_report.json, rapport_final_pipeline.json,
params_sir_calibrated.gaml) correspondent donc a la Sierra Leone, comme
dans resultats_reference/.
=============================================================================
"""

import argparse
import os
import subprocess
import sys
import time

# (nom, fichier de donnees, population N, maladie, D_I en jours)
ENTITES = [
    ("New York",     "data/usa_county_wise.csv",    19_450_000, "COVID-19",    14),
    ("California",   "data/usa_county_wise.csv",    39_538_223, "COVID-19",    14),
    ("Florida",      "data/usa_county_wise.csv",    21_538_187, "COVID-19",    14),
    ("Texas",        "data/usa_county_wise.csv",    29_145_505, "COVID-19",    14),
    ("Guinea",       "data/ebola_guinea.csv",       11_474_383, "Ebola (EVD)", 7),
    ("Liberia",      "data/ebola_liberia.csv",       4_092_310, "Ebola (EVD)", 7),
    ("Sierra Leone", "data/ebola_sierra_leone.csv",  5_743_725, "Ebola (EVD)", 7),
]


def environnement_sans_ia():
    """Le prior M2 des resultats de reference vient du repli litterature
    (aucune cle API). Une cle presente dans .env ferait interroger un LLM,
    dont la reponse n'est pas deterministe : on la neutralise en definissant
    les variables a vide, que load_dotenv() n'ecrase pas."""
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = ""
    env["GEMINI_API_KEY"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    return env


def lancer(script, nom, data, N, maladie, D_I):
    cmd = [sys.executable, script,
           "--data", data, "--state", nom, "--N", str(N),
           "--disease", maladie, "--D-I", str(D_I)]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, env=environnement_sans_ia())
    print(f"<<< {script} termine en {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="nom d'une entite (ex. \"New York\")")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="ne lance que M6_comparaison.py")
    args = ap.parse_args()

    # Les scripts utilisent des chemins relatifs (data/, outputs/)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    entites = ENTITES
    if args.only:
        entites = [e for e in ENTITES if e[0].lower() == args.only.lower()]
        if not entites:
            sys.exit(f"Entite inconnue : {args.only}. "
                     f"Choix : {', '.join(e[0] for e in ENTITES)}")

    os.makedirs("outputs", exist_ok=True)
    t0 = time.time()
    for nom, data, N, maladie, D_I in entites:
        print("\n" + "#" * 70 + f"\n#  {nom}  (N = {N:,}, {maladie}, D_I = {D_I} j)\n" + "#" * 70)
        if not args.skip_pipeline:
            lancer("pipeline.py", nom, data, N, maladie, D_I)
        lancer("M6_comparaison.py", nom, data, N, maladie, D_I)

    print(f"\nTermine en {(time.time() - t0) / 60:.1f} min. "
          "Verification : python verifier_resultats.py")


if __name__ == "__main__":
    main()
