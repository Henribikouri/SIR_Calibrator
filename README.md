# sir_calibrator

Pipeline d'estimation automatique des paramètres d'un modèle épidémiologique
SIR par calibration séquentielle θ(t), avec génération d'un calendrier de
paramètres pour la plateforme multi-agent GAMA.

Mémoire de Master 2 Génie Logiciel, Henri BIKOURI, ENSPY / Université de
Yaoundé I.

Ce dépôt contient le code, les données d'entrée et les résultats de référence
du mémoire. Les commandes ci-dessous régénèrent les résultats des 7 entités
étudiées et les comparent automatiquement aux résultats de référence.

## Résultats de référence

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
Tous les fichiers correspondants se trouvent dans
[resultats_reference/](resultats_reference/).

## Reproduire les résultats

### 1. Prérequis

- **Python 3.12** (les versions 3.10 et 3.11 conviennent aussi ; la 3.13 n'est
  pas prise en charge par numpy 1.26.4). Vérifier avec `python --version`.
- Git.
- Environ 1 Go d'espace disque (environnement virtuel compris).

### 2. Cloner et installer

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

### 3. Exécuter

```bash
python reproduire.py
```

Le script lance, pour chacune des 7 entités, `pipeline.py` (modules M1 à M5)
puis `M6_comparaison.py`, avec les arguments exacts des résultats de
référence. Durée : 5 à 10 minutes selon la machine. Les sorties sont écrites
dans `outputs/`.

### 4. Vérifier

```bash
python verifier_resultats.py
```

Chaque CSV, JSON et fichier GAML de `resultats_reference/` est comparé à son
homologue de `outputs/` (les horodatages sont ignorés). La dernière ligne
indique le nombre de fichiers conformes ; le code de sortie vaut 0 si tout
concorde.

Les résultats ont été reproduits à l'identique, octet pour octet, dans un
environnement neuf installé depuis `requirements.txt` (Windows 11,
Python 3.12.6).

## Lancer une entité à la main

```bash
# COVID-19, données Johns Hopkins CSSE (D_I = 14 jours, valeur par défaut)
python pipeline.py       --state "California" --N 39538223
python M6_comparaison.py --state "California" --N 39538223

# Ebola, données OMS (D_I = 7 jours)
python pipeline.py       --data data/ebola_guinea.csv --state "Guinea" --N 11474383 --disease "Ebola (EVD)" --D-I 7
python M6_comparaison.py --data data/ebola_guinea.csv --state "Guinea" --N 11474383 --disease "Ebola (EVD)" --D-I 7
```

Une seule entité via le script de reproduction :
`python reproduire.py --only "Texas"`.

| Argument | Rôle | Défaut |
|---|---|---|
| `--data` | CSV d'entrée | `data/usa_county_wise.csv` |
| `--state` | Entité (valeur de `Province_State` pour les données JHU) | `New York` |
| `--N` | Population totale | `19450000` |
| `--disease` | Maladie (choisit le prior de repli de M2) | `COVID-19` |
| `--mode` | Format du CSV : `auto`, `jhu`, `gama`, `generic` | `auto` |
| `--D-I` | Durée infectieuse a priori, en jours | `14` |
| `--sobol` | Échantillons de Sobol, `pipeline.py` seulement | `256` |
| `--no-plot` | Sans figures, `pipeline.py` seulement | désactivé |
| `--outdir` | Dossier de sortie, `M6_comparaison.py` seulement | `outputs` |

La liste des 58 entités du jeu de données américain, avec leur population
(recensement 2020), figure dans [data/RECAP_ETATS.md](data/RECAP_ETATS.md).

## Architecture

| Module | Fichier | Rôle |
|---|---|---|
| M1 | `M1_acquisition_cameroun.py` | Lecture du CSV (détection du format), reconstruction de S, I, R, lissage de Savitzky-Golay |
| M2 | `M2_ia_prior.py` | Point de départ θ₀ : LLM (Claude, puis Gemini) si une clé API est fournie, sinon valeurs de la littérature |
| M3 | `M3_calibration.py` | Calibration séquentielle θ(t) par fenêtres glissantes chaînées (évolution différentielle puis Nelder-Mead) |
| M4 | `M4_generation_gaml.py` | Calendrier de paramètres par morceaux pour GAMA (`.gaml`) |
| M5 | `M5_validation.py` | Validation prospective et analyse de sensibilité de Sobol |
| M6 | `M6_comparaison.py` | Rejoue M1 à M3 et exporte les jeux de comparaison et les figures FIG1 à FIG3 |

`pipeline.py` orchestre M1 à M5. `M1_acquisition.py` est l'ancienne version de
M1, limitée au format JHU, utilisée seulement si la version universelle est
absente. `wave_selection.py` n'est plus appelé par le pipeline ; il est
conservé pour mémoire. La méthode est détaillée dans
[METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md](METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md).

## Reproductibilité

- **M3** fixe la graine de l'évolution différentielle (`WIN_DE_SEED = 42`) :
  deux exécutions donnent la même trajectoire θ(t).
- **M2** n'interroge un LLM que si `ANTHROPIC_API_KEY` ou `GEMINI_API_KEY`
  est définie (fichier `.env`, voir [.env.example](.env.example)). La réponse
  d'un LLM n'étant pas déterministe, les résultats de référence utilisent le
  repli littérature (OMS/CDC pour la COVID-19, Althaus 2014 et WHO Ebola
  Response Team 2014 pour Ebola). `reproduire.py` neutralise toute clé
  présente.
- **Versions figées** dans `requirements.txt` : l'optimiseur et le solveur
  d'EDO (`odeint`) dépendent des versions de numpy et scipy. Sur un autre
  système d'exploitation, des écarts d'arrondi très faibles restent possibles ;
  `verifier_resultats.py` tolère un écart relatif de 10⁻⁶.
- Les fichiers sans suffixe d'entité (`calibration_report.json`,
  `rapport_final_pipeline.json`, `params_sir_calibrated.gaml`,
  `COVID19_NewYork.sir`) sont écrasés à chaque exécution de `pipeline.py` ;
  ceux de référence correspondent à la Sierra Leone, dernière entité traitée.
- Les figures PNG ne sont pas comparées octet par octet, leur rendu dépendant
  de la plateforme.

## Données

| Fichier | Contenu | Source |
|---|---|---|
| `data/usa_county_wise.csv` | Cas confirmés et décès COVID-19 par comté, 22 janvier au 27 juillet 2020 | Johns Hopkins University CSSE |
| `data/ebola_data_db_format.csv` | Indicateurs Ebola par pays, 29 août 2014 au 23 mars 2016 | Organisation mondiale de la santé |
| `data/ebola_guinea.csv`, `ebola_liberia.csv`, `ebola_sierra_leone.csv` | Cas et décès cumulés (confirmés, probables et suspects) par pays, extraits du fichier précédent, 258 relevés jusqu'au 29 décembre 2015 | Organisation mondiale de la santé |
