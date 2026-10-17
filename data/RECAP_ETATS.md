# Recapitulatif des etats du dataset `usa_county_wise.csv`

> Source : Johns Hopkins CSSE -- COVID-19 USA (niveau *county*).  
> Periode couverte : **2020-01-22 -> 2020-07-27** (188 jours).  
> Granularite : **58 entites** (50 etats + DC + 5 territoires + 2 navires de croisiere).  
> Populations : **US Census 2020** (habitants), utilisees comme $N$ dans le modele SIR.

Ce document sert de **catalogue de simulation** : chaque ligne donne le nom exact `Province_State` 
a passer en `TARGET_STATE`, la population $N$ a passer en `N_POPULATION`, et un indicateur de qualite pour la calibration.

## Tableau complet

| # | Etat / Territoire | Population $N$ (2020) | Comtes | Jours | Cas cumules (fin) | Deces (fin) | Taux d'attaque | CFR | Aptitude SIR |
|--:|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| 1 | California | 39,538,223 | 60 | 188 | 458,121 | 8,494 | 1.16% | 1.85% | Excellente |
| 2 | Florida | 21,538,187 | 69 | 188 | 432,747 | 5,941 | 2.01% | 1.37% | Excellente |
| 3 | New York | 20,201,249 | 64 | 188 | 412,344 | 32,645 | 2.04% | 7.92% | Excellente |
| 4 | Texas | 29,145,505 | 256 | 188 | 400,336 | 5,633 | 1.37% | 1.41% | Excellente |
| 5 | New Jersey | 9,288,994 | 23 | 188 | 179,812 | 15,804 | 1.94% | 8.79% | Excellente |
| 6 | Illinois | 12,812,508 | 104 | 188 | 173,894 | 7,417 | 1.36% | 4.27% | Excellente |
| 7 | Georgia | 10,711,908 | 161 | 188 | 170,843 | 3,429 | 1.59% | 2.01% | Excellente |
| 8 | Arizona | 7,151,502 | 17 | 188 | 163,827 | 3,349 | 2.29% | 2.04% | Excellente |
| 9 | Massachusetts | 7,029,917 | 17 | 188 | 115,926 | 8,553 | 1.65% | 7.38% | Excellente |
| 10 | North Carolina | 10,439,388 | 102 | 188 | 114,916 | 1,838 | 1.10% | 1.60% | Excellente |
| 11 | Pennsylvania | 13,002,700 | 69 | 188 | 112,995 | 7,132 | 0.87% | 6.31% | Excellente |
| 12 | Louisiana | 4,657,757 | 66 | 188 | 109,917 | 3,675 | 2.36% | 3.34% | Excellente |
| 13 | Tennessee | 6,910,840 | 97 | 188 | 96,489 | 962 | 1.40% | 1.00% | Excellente |
| 14 | Michigan | 10,077,331 | 87 | 188 | 87,173 | 6,420 | 0.87% | 7.36% | Excellente |
| 15 | Virginia | 8,631,393 | 135 | 188 | 86,072 | 2,082 | 1.00% | 2.42% | Excellente |
| 16 | Ohio | 11,799,448 | 90 | 188 | 85,177 | 3,344 | 0.72% | 3.93% | Excellente |
| 17 | Maryland | 6,177,224 | 26 | 188 | 84,876 | 3,544 | 1.37% | 4.18% | Excellente |
| 18 | South Carolina | 5,118,425 | 48 | 188 | 82,417 | 1,707 | 1.61% | 2.07% | Excellente |
| 19 | Alabama | 5,024,279 | 69 | 188 | 81,115 | 1,447 | 1.61% | 1.78% | Excellente |
| 20 | Indiana | 6,785,528 | 94 | 188 | 62,907 | 2,709 | 0.93% | 4.31% | Excellente |
| 21 | Washington | 7,705,281 | 41 | 188 | 53,321 | 1,518 | 0.69% | 2.85% | Excellente |
| 22 | Mississippi | 2,961,279 | 84 | 188 | 52,781 | 1,498 | 1.78% | 2.84% | Excellente |
| 23 | Minnesota | 5,706,494 | 89 | 188 | 51,803 | 1,582 | 0.91% | 3.05% | Excellente |
| 24 | Wisconsin | 5,893,718 | 74 | 188 | 49,417 | 893 | 0.84% | 1.81% | Excellente |
| 25 | Connecticut | 3,605,944 | 10 | 188 | 48,983 | 4,418 | 1.36% | 9.02% | Excellente |
| 26 | Colorado | 5,773,714 | 66 | 188 | 44,547 | 1,799 | 0.77% | 4.04% | Excellente |
| 27 | Nevada | 3,104,614 | 19 | 188 | 43,831 | 739 | 1.41% | 1.69% | Excellente |
| 28 | Missouri | 6,154,913 | 118 | 188 | 43,489 | 1,218 | 0.71% | 2.80% | Excellente |
| 29 | Iowa | 3,190,369 | 101 | 188 | 42,696 | 1,028 | 1.34% | 2.41% | Excellente |
| 30 | Arkansas | 3,011,524 | 77 | 188 | 39,447 | 408 | 1.31% | 1.03% | Bonne |
| 31 | Utah | 3,271,616 | 37 | 188 | 38,409 | 297 | 1.17% | 0.77% | Bonne |
| 32 | Oklahoma | 3,959,353 | 79 | 188 | 32,685 | 495 | 0.83% | 1.51% | Bonne |
| 33 | Kentucky | 4,505,836 | 122 | 188 | 27,586 | 709 | 0.61% | 2.57% | Bonne |
| 34 | Kansas | 2,937,880 | 107 | 188 | 25,804 | 529 | 0.88% | 2.05% | Bonne |
| 35 | Nebraska | 1,961,504 | 95 | 188 | 24,899 | 317 | 1.27% | 1.27% | Bonne |
| 36 | New Mexico | 2,117,522 | 35 | 188 | 19,502 | 619 | 0.92% | 3.17% | Bonne |
| 37 | Idaho | 1,839,106 | 46 | 188 | 18,692 | 232 | 1.02% | 1.24% | Bonne |
| 38 | Rhode Island | 1,097,379 | 7 | 188 | 18,515 | 0 | 1.69% | 0.00% | Bonne |
| 39 | Oregon | 4,237,256 | 38 | 188 | 17,088 | 289 | 0.40% | 1.69% | Bonne |
| 40 | Puerto Rico | 3,285,874 | 80 | 188 | 15,431 | 0 | 0.47% | 0.00% | Bonne |
| 41 | Delaware | 989,948 | 5 | 188 | 14,406 | 569 | 1.46% | 3.95% | Moyenne |
| 42 | District of Columbia | 689,545 | 3 | 188 | 11,780 | 581 | 1.71% | 4.93% | Moyenne |
| 43 | South Dakota | 886,667 | 68 | 188 | 8,444 | 1,126 | 0.95% | 13.33% | Moyenne |
| 44 | New Hampshire | 1,377,529 | 12 | 188 | 6,441 | 409 | 0.47% | 6.35% | Moyenne |
| 45 | West Virginia | 1,793,716 | 57 | 188 | 5,999 | 104 | 0.33% | 1.73% | Moyenne |
| 46 | North Dakota | 779,094 | 55 | 188 | 5,986 | 99 | 0.77% | 1.65% | Moyenne |
| 47 | Maine | 1,362,359 | 18 | 188 | 3,832 | 119 | 0.28% | 3.11% | Moyenne |
| 48 | Montana | 1,084,225 | 58 | 188 | 3,380 | 47 | 0.31% | 1.39% | Moyenne |
| 49 | Alaska | 733,391 | 31 | 188 | 2,623 | 67 | 0.36% | 2.55% | Faible |
| 50 | Wyoming | 576,851 | 25 | 188 | 2,520 | 1 | 0.44% | 0.04% | Faible |
| 51 | Hawaii | 1,455,271 | 7 | 188 | 1,711 | 25 | 0.12% | 1.46% | Faible |
| 52 | Vermont | 643,077 | 16 | 188 | 1,402 | 137 | 0.22% | 9.77% | Faible |
| 53 | Virgin Islands | 106,290 | 1 | 188 | 364 | 7 | 0.34% | 1.92% | Faible |
| 54 | Guam | 168,775 | 1 | 188 | 349 | 5 | 0.21% | 1.43% | Faible |
| 55 | Grand Princess | 3,533 | 1 | 188 | 103 | 0 | 2.92% | 0.00% | Faible |
| 56 | Diamond Princess | 3,711 | 1 | 188 | 49 | 0 | 1.32% | 0.00% | Faible |
| 57 | Northern Mariana Islands | 47,329 | 1 | 188 | 40 | 2 | 0.08% | 5.00% | Faible |
| 58 | American Samoa | 49,710 | 1 | 188 | 0 | 0 | 0.00% | 0.00% | Faible |

## Regions recommandees pour la calibration (Aptitude "Excellente")

Population importante, nombreux comtes et vague epidemique nette -- ideales pour viser la mention excellente sur plusieurs contrees.

```python
# Liste (TARGET_STATE, N_POPULATION) prete a boucler
REGIONS_EXCELLENTES = [
    ("California", 39538223),
    ("Florida", 21538187),
    ("New York", 20201249),
    ("Texas", 29145505),
    ("New Jersey", 9288994),
    ("Illinois", 12812508),
    ("Georgia", 10711908),
    ("Arizona", 7151502),
    ("Massachusetts", 7029917),
    ("North Carolina", 10439388),
    ("Pennsylvania", 13002700),
    ("Louisiana", 4657757),
    ("Tennessee", 6910840),
    ("Michigan", 10077331),
    ("Virginia", 8631393),
    ("Ohio", 11799448),
    ("Maryland", 6177224),
    ("South Carolina", 5118425),
    ("Alabama", 5024279),
    ("Indiana", 6785528),
    ("Washington", 7705281),
    ("Mississippi", 2961279),
    ("Minnesota", 5706494),
    ("Wisconsin", 5893718),
    ("Connecticut", 3605944),
    ("Colorado", 5773714),
    ("Nevada", 3104614),
    ("Missouri", 6154913),
    ("Iowa", 3190369),
]
```

## Notes methodologiques

- **Taux d'attaque** = cas cumules finaux / population x 100. En dessous de ~0,05 %, le signal est trop faible pour identifier beta et gamma de facon fiable.
- **CFR** (Case Fatality Ratio) = deces / cas cumules. Certains territoires (Puerto Rico, Rhode Island) reportent 0 deces -> donnees deces incompletes, a eviter pour une calibration sur les deces.
- **Navires** (`Diamond Princess`, `Grand Princess`) et **territoires mono-comte** (`Guam`, `American Samoa`, ...) : population trop petite ou dynamique atypique -> exclus des recommandations.
- Les populations sont des **estimations Census 2020** ; ajuster si une valeur exacte au jour de la simulation est requise.