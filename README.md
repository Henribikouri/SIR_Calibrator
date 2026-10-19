# sir_calibrator

Pipeline d'estimation automatique des paramètres d'un modèle épidémiologique
SIR par calibration séquentielle θ(t), avec génération d'un calendrier de
paramètres pour la plateforme multi-agent GAMA.

Mémoire de Master 2 Génie Logiciel, Henri BIKOURI, ENSPY / Université de
Yaoundé I.

Le pipeline accepte n'importe quel jeu de données épidémiologiques au format
CSV (Johns Hopkins, OMS, sortie de simulation GAMA, fichier générique) et
n'importe quelle région, population ou maladie.

## Installation

Prérequis : **Python 3.10, 3.11 ou 3.12** (pas 3.13, non pris en charge par
numpy 1.26.4) et Git. Vérifier avec `python --version`.

Windows (PowerShell) :

```powershell
git clone https://github.com/Henribikouri/SIR_Calibrator.git
cd SIR_Calibrator
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell refuse d'activer l'environnement, exécuter d'abord
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Linux / macOS :

```bash
git clone https://github.com/Henribikouri/SIR_Calibrator.git
cd SIR_Calibrator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Utilisation

Le projet a deux points d'entrée, à lancer avec les mêmes arguments :

| Script | Modules | Sorties dans `outputs/` |
|---|---|---|
| `python pipeline.py` | M1 à M5 | Figures `M1_*`, `M2_*`, `M3_*`, `M5_*`, calendrier GAMA `params_sir_calibrated.gaml`, rapports JSON |
| `python M6_comparaison.py` | M1 à M3 rejoués, puis comparaison | CSV `dataset_compare_*`, `theta_trajectoire_*`, `dynamique_temporelle_*`, figures `FIG1` à `FIG3` |

```bash
# Valeurs par défaut : État de New York, données JHU
python pipeline.py

# Un autre État américain (liste et populations dans data/RECAP_ETATS.md)
python pipeline.py       --state "Georgia" --N 10711908
python M6_comparaison.py --state "Georgia" --N 10711908

# Un autre jeu de données et une autre maladie
python pipeline.py --data data/ebola_guinea.csv --state "Guinea" --N 11474383 --disease "Ebola (EVD)" --D-I 7

# Une sortie de simulation GAMA
python pipeline.py --data mon_export_gama.csv --state "Cameroun" --N 5000 --mode gama

# Exécution rapide, sans figures
python pipeline.py --state "Texas" --N 29145505 --no-plot
```

### Arguments

| Argument | Rôle | Défaut |
|---|---|---|
| `--data` | Chemin du CSV d'entrée | `data/usa_county_wise.csv` |
| `--state` | Région (pour un CSV JHU : valeur de la colonne `Province_State`) | `New York` |
| `--N` | Population totale de la région | `19450000` |
| `--disease` | Maladie ; si le nom contient « Ebola », M2 prend les valeurs de repli Ebola, sinon celles de la COVID-19 | `COVID-19` |
| `--mode` | Format du CSV : `auto`, `jhu`, `gama`, `generic` | `auto` |
| `--D-I` | Durée infectieuse a priori, en jours | `14` |
| `--sobol` | Échantillons de l'analyse de Sobol (`pipeline.py`) | `256` |
| `--no-plot` | Désactive les figures (`pipeline.py`) | désactivé |
| `--outdir` | Dossier de sortie (`M6_comparaison.py`) | `outputs` |

### Formats de CSV acceptés

Avec `--mode auto`, M1 reconnaît le format d'après les noms de colonnes :

| Format | Colonnes attendues | Exemple |
|---|---|---|
| GAMA | `tick`, `S_total`, `I_total`, `R_total` | Export d'une simulation GAMA |
| JHU | `Date`, `Confirmed`, et éventuellement `Province_State` (filtrée par `--state`, sommée par date) | `data/usa_county_wise.csv`, `data/ebola_*.csv` |
| Générique | Une colonne de temps (`date`, `day`, `jour`, `week`...) et une colonne de cas (`cases`, `cas`, `confirmed`, `infected`...), cumulés ou journaliers | Données d'un ministère, de l'OMS, de Kaggle |

### Prior par IA générative (optionnel)

M2 peut demander le point de départ θ₀ à un LLM (Claude, puis Gemini). Pour
l'activer, copier [.env.example](.env.example) en `.env` et y renseigner
`ANTHROPIC_API_KEY` ou `GEMINI_API_KEY`. Sans clé, M2 utilise des valeurs de
la littérature (OMS/CDC pour la COVID-19, Althaus 2014 et WHO Ebola Response
Team 2014 pour Ebola).

### Remarques

- Les fichiers `calibration_report.json`, `rapport_final_pipeline.json`,
  `params_sir_calibrated.gaml` et `*.sir` n'ont pas de suffixe de région :
  chaque exécution de `pipeline.py` écrase ceux de la précédente. Les autres
  sorties portent le suffixe `<Région>_<N>`.
- M3 fixe la graine de l'optimiseur (`WIN_DE_SEED = 42`) : deux exécutions
  avec les mêmes arguments donnent les mêmes résultats.

## Architecture

| Module | Fichier | Rôle |
|---|---|---|
| M1 | `M1_acquisition_cameroun.py` | Lecture du CSV (détection du format), reconstruction de S, I, R, lissage de Savitzky-Golay |
| M2 | `M2_ia_prior.py` | Point de départ θ₀ : LLM si une clé API est fournie, sinon valeurs de la littérature |
| M3 | `M3_calibration.py` | Calibration séquentielle θ(t) par fenêtres glissantes chaînées (évolution différentielle puis Nelder-Mead) |
| M4 | `M4_generation_gaml.py` | Calendrier de paramètres par morceaux pour GAMA (`.gaml`) |
| M5 | `M5_validation.py` | Validation prospective et analyse de sensibilité de Sobol |
| M6 | `M6_comparaison.py` | Rejoue M1 à M3 et exporte les jeux de comparaison et les figures FIG1 à FIG3 |

`pipeline.py` orchestre M1 à M5. `M1_acquisition.py` est l'ancienne version de
M1, limitée au format JHU, utilisée seulement si la version universelle est
absente. `wave_selection.py` n'est plus appelé par le pipeline ; il est
conservé pour mémoire. La méthode est détaillée dans
[METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md](METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md).

## Données fournies

| Fichier | Contenu | Source |
|---|---|---|
| `data/usa_county_wise.csv` | Cas confirmés et décès COVID-19 par comté, 22 janvier au 27 juillet 2020, 58 entités | Johns Hopkins University CSSE |
| `data/RECAP_ETATS.md` | Liste des 58 entités avec leur population (recensement 2020) | US Census Bureau |
| `data/ebola_data_db_format.csv` | Indicateurs Ebola par pays, 29 août 2014 au 23 mars 2016 | Organisation mondiale de la santé |
| `data/ebola_guinea.csv`, `ebola_liberia.csv`, `ebola_sierra_leone.csv` | Cas et décès cumulés (confirmés, probables et suspects) par pays, extraits du fichier précédent, 258 relevés jusqu'au 29 décembre 2015 | Organisation mondiale de la santé |

## Vérifier les résultats du mémoire (facultatif)

Le dossier [resultats_reference/](resultats_reference/) contient les sorties
des 7 entités étudiées dans le mémoire :

| Entité | Maladie | N | Jours | R²_I | R²_C | RMSE (cas) |
|---|---|--:|--:|--:|--:|--:|
| New York | COVID-19 | 19 450 000 | 188 | 0,9011 | 0,9957 | 12 412,92 |
| Californie | COVID-19 | 39 538 223 | 188 | 0,9963 | 0,9989 | 2 248,64 |
| Floride | COVID-19 | 21 538 187 | 188 | 0,9919 | 0,9974 | 4 046,31 |
| Texas | COVID-19 | 29 145 505 | 188 | 0,9945 | 0,9982 | 2 899,74 |
| Guinée | Ebola | 11 474 383 | 258 | 0,9341 | 0,9973 | 37,61 |
| Libéria | Ebola | 4 092 310 | 258 | 0,8902 | 0,9955 | 163,69 |
| Sierra Leone | Ebola | 5 743 725 | 258 | 0,9553 | 0,9979 | 129,12 |

R²_I porte sur la prévalence I(t), R²_C sur le cumul C(t) = I(t) + R(t).

Pour les régénérer et les comparer automatiquement :

```bash
python reproduire.py          # lance pipeline.py et M6_comparaison.py pour les 7 entités (5 à 10 min)
python verifier_resultats.py  # compare outputs/ à resultats_reference/
```

`reproduire.py` ne fait qu'appeler les deux scripts ci-dessus avec les
arguments du mémoire. Il ignore les clés API éventuelles, car la réponse
d'un LLM n'est pas déterministe et les résultats du mémoire ont été obtenus
avec les valeurs de la littérature. `verifier_resultats.py` compare chaque
CSV, JSON et fichier GAML (horodatages exclus) et doit afficher `66/66`.
Vérifié dans un environnement neuf installé depuis `requirements.txt`
(Windows 11, Python 3.12) : résultats identiques octet pour octet.
