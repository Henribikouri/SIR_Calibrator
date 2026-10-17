"""
=============================================================================
  MODULE WAVE_SELECTION -- FENETRAGE SUR LA PREMIERE VAGUE EPIDEMIQUE
=============================================================================

  >>> DEPRECIE POUR LA CALIBRATION -- NON APPELE PAR LE PIPELINE ACTUEL <<<

  Depuis la refonte "calibration sequentielle theta(t)" (voir
  METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md), M3_calibration.py calibre par
  fenetres glissantes CHAINEES sur TOUTE la serie observee et la population
  COMPLETE N : un theta(t) qui varie dans le temps capture nativement les
  dynamiques multi-vagues, sans qu'il soit necessaire d'isoler et de decouper
  au prealable la premiere vague. pipeline.py et M6_comparaison.py n'appellent
  donc plus select_first_wave().

  Ce fichier est CONSERVE (non supprime) car son algorithme de detection de
  pic (detect_first_wave) reste potentiellement utile ailleurs, par exemple
  pour identifier automatiquement la fenetre theta_k "du pic" utilisee par
  l'analyse de sensibilite de Sobol dans M5_validation.py.

  Pipeline d'estimation automatique des parametres SIR
  Auteur : Henri BIKOURI

  Role (Priorite 1 de la revue -- version historique, avant la refonte
  sequentielle) :
  --------------------------------
  Un modele SIR a parametres constants (alpha, gamma) est mathematiquement
  contraint a produire UNE SEULE cloche epidemique. Calibrer un tel modele
  sur une serie multi-vagues (ex. Californie janv->juil 2020 : plateau + vague
  estivale) donne un ajustement mediocre (R2 ~ 0.66) et une validation croisee
  qui diverge.

  Ce module isole automatiquement la PREMIERE vague epidemique bien formee
  (montee -> pic -> debut de declin, avant le creux qui precede une eventuelle
  seconde vague). En calibrant le SIR sur cette fenetre, le modele redevient
  le bon modele -> R2 attendu > 0.95, RMSE faible, validation croisee saine.

  Comportement honnete :
  ----------------------
  Si AUCUNE premiere vague isolee n'existe (serie monotone ou plateau domine
  par une vague ulterieure -- cas Californie / Floride / Texas), le detecteur
  retourne None. Le pipeline retombe alors sur la serie complete et signale
  qu'il faut la Priorite 2 (transmission variable beta(t)) pour cette region.

=============================================================================
"""

import numpy as np
from scipy.signal import find_peaks


# -----------------------------------------------------------------------------
# PARAMETRES DU DETECTEUR (valeurs par defaut, surchargeable)
# -----------------------------------------------------------------------------
PROM_FRAC    = 0.15   # Proeminence minimale d'un pic (fraction du max de la serie)
MIN_DISTANCE = 14     # Distance minimale entre pics/creux (jours) -- 1 cycle infectieux
ONSET_FRAC   = 0.02   # Debut de fenetre : ou I(t) depasse 2% du pic (retire les zeros initiaux)
DECLINE_FRAC = 0.60   # Fin de fenetre : creux inter-vagues tombe sous 60% du pic
MIN_LEN      = 30     # Longueur minimale d'une fenetre exploitable (jours)


# -----------------------------------------------------------------------------
# DETECTION DE LA PREMIERE VAGUE
# -----------------------------------------------------------------------------
def detect_first_wave(I,
                      prom_frac=PROM_FRAC,
                      distance=MIN_DISTANCE,
                      onset_frac=ONSET_FRAC,
                      decline_frac=DECLINE_FRAC,
                      min_len=MIN_LEN):
    """
    Detecte la fenetre [start, end] de la premiere vague epidemique.

    Algorithme :
      1. Cherche les pics proeminents de I(t) (find_peaks, proeminence
         relative au max global) -> le PREMIER pic = sommet de la 1re vague.
      2. onset (start)  : en remontant depuis le pic, premier jour ou I(t)
         retombe sous onset_frac * I_pic (borne basse de la montee).
      3. offset (end)   : premier creux APRES le pic qui descend sous
         decline_frac * I_pic (le val inter-vagues). Sinon fin de serie.

    Parameters
    ----------
    I            : array-like -- serie des infectes actifs I(t) (lissee)
    prom_frac    : float -- proeminence min d'un pic (x max de la serie)
    distance     : int   -- ecart min entre extrema (jours)
    onset_frac   : float -- seuil de debut de vague (x pic)
    decline_frac : float -- seuil du creux inter-vagues (x pic)
    min_len      : int   -- longueur minimale acceptable de la fenetre

    Returns
    -------
    dict | None
        {'start', 'peak', 'end', 'I_peak', 'reason'} si une vague isolee
        existe, sinon None (avec la raison loggee).
    """
    I = np.asarray(I, dtype=float)
    T = len(I)
    if T < min_len:
        return None
    Imax = I.max()
    if Imax <= 0:
        return None

    # --- 1. Pic de la premiere vague -------------------------------------
    peaks, props = find_peaks(I, prominence=prom_frac * Imax, distance=distance)
    if len(peaks) == 0:
        # Aucun pic interne proeminent : serie monotone / plateau domine par
        # une vague ulterieure (I(t) grimpe jusqu'au bord). Pas de vague isolee.
        return None
    peak = int(peaks[0])
    I_peak = float(I[peak])

    # Un pic trop proche du bord droit = vague non terminee (encore en montee)
    if peak >= T - distance:
        return None

    # --- 2. Debut de la fenetre (onset) ----------------------------------
    start = 0
    thr_lo = onset_frac * I_peak
    for i in range(peak, -1, -1):
        if I[i] <= thr_lo:
            start = i
            break

    # --- 3. Fin de la fenetre (creux inter-vagues ou fin de serie) -------
    troughs, _ = find_peaks(-I, distance=distance)
    troughs = [int(t) for t in troughs if t > peak]
    end = T - 1
    reason = "declin jusqu'a la fin de serie (vague unique propre)"
    for tr in troughs:
        if I[tr] <= decline_frac * I_peak:
            end = tr
            reason = "creux inter-vagues detecte (fenetrage avant 2e vague)"
            break

    if (end - start + 1) < min_len:
        return None

    return {
        'start':  int(start),
        'peak':   int(peak),
        'end':    int(end),
        'I_peak': I_peak,
        'reason': reason,
    }


# -----------------------------------------------------------------------------
# APPLICATION DU FENETRAGE A (df_sir, meta)
# -----------------------------------------------------------------------------
def select_first_wave(df_sir, meta, enabled=True, verbose=True, **kwargs):
    """
    Applique le fenetrage 1re vague sur les sorties de M1.

    Decoupe df_sir et meta de facon coherente sur [start, end], reindexe le
    temps a 0, et met a jour T / peak_I / dates. Si aucune vague isolee n'est
    detectee (ou si enabled=False), retourne les donnees inchangees avec un
    drapeau explicite.

    Parameters
    ----------
    df_sir  : pd.DataFrame -- sortie de M1 (colonnes t, Date, S/I/R_data, ...)
    meta    : dict         -- metadonnees de M1 (arrays t_obs/S_data/I_data/R_data)
    enabled : bool         -- activer le fenetrage (defaut True)
    verbose : bool         -- afficher le rapport de detection

    Returns
    -------
    (df_w, meta_w, wave_info) : DataFrame + dict fenetres + dict d'info
        wave_info contient au minimum :
          'applied'  : bool  -- True si un fenetrage a ete applique
          'start','end','peak','n_days','I_peak','reason'
    """
    I = np.asarray(meta['I_data'], dtype=float)
    T0 = len(I)

    if verbose:
        print("\n" + "=" * 65)
        print("  FENETRAGE PRIORITE 1 -- Selection de la 1re vague epidemique")
        print("=" * 65)

    if not enabled:
        if verbose:
            print("  -> Desactive (--wave full) : calibration sur la serie complete.")
        return df_sir, meta, {'applied': False, 'reason': 'desactive'}

    win = detect_first_wave(I, **kwargs)

    if win is None:
        if verbose:
            print("  -> Aucune 1re vague isolee detectee (serie monotone /")
            print("     plateau domine par une vague ulterieure).")
            print("     ==> Calibration sur serie complete. Region candidate a la")
            print("         Priorite 2 (transmission variable beta(t)).")
        return df_sir, meta, {
            'applied': False,
            'reason': 'aucune vague isolee (candidat Priorite 2)',
        }

    s, e, p = win['start'], win['end'], win['peak']
    n_days = e - s + 1

    # --- Decoupe coherente du DataFrame ----------------------------------
    df_w = df_sir.iloc[s:e + 1].reset_index(drop=True).copy()
    df_w['t'] = np.arange(len(df_w))   # temps reindexe a 0
    has_date = 'Date' in df_w.columns  # certaines versions de M1 n'ont pas de Date

    # --- Decoupe coherente des metadonnees -------------------------------
    meta_w = dict(meta)  # copie superficielle
    for key in ('t_obs', 'S_data', 'I_data', 'R_data'):
        meta_w[key] = np.asarray(meta[key])[s:e + 1]
    meta_w['t_obs']      = np.arange(n_days, dtype=float)   # reindexe a 0
    meta_w['T']          = n_days
    meta_w['peak_I']     = float(np.max(meta_w['I_data']))
    if has_date:
        meta_w['date_start'] = df_w['Date'].iloc[0]
        meta_w['date_end']   = df_w['Date'].iloc[-1]
    meta_w['wave_window'] = {'start': s, 'end': e, 'peak': p}

    if verbose:
        print(f"  -> Vague isolee : {win['reason']}")
        print(f"  -> Fenetre      : J{s} -> J{e}  ({n_days} jours) sur {T0} initiaux")
        if has_date:
            d0 = str(df_w['Date'].iloc[0])[:10]
            d1 = str(df_w['Date'].iloc[-1])[:10]
            print(f"  -> Dates        : {d0} -> {d1}")
        print(f"  -> Pic 1re vague: J{p - s} (reindexe) | I_pic = {win['I_peak']:,.0f}")
        print(f"  -> Reduction    : {T0} -> {n_days} points "
              f"({100*(1-n_days/T0):.0f}% de tail multi-vagues retiree)")
    print("=" * 65)

    wave_info = {
        'applied': True,
        'start':  s,
        'end':    e,
        'peak':   p,
        'n_days': n_days,
        'I_peak': win['I_peak'],
        'reason': win['reason'],
    }
    return df_w, meta_w, wave_info


# -----------------------------------------------------------------------------
# TEST AUTONOME
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from scipy.signal import savgol_filter

    df = pd.read_csv('data/usa_county_wise.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

    def recon(state, D_I=14):
        d = (df[df['Province_State'] == state]
             .groupby('Date').agg(c=('Confirmed', 'sum'))
             .reset_index().sort_values('Date'))
        new = d['c'].diff().fillna(0).clip(lower=0)
        I_raw = new.rolling(window=D_I, min_periods=1).sum().values
        return np.clip(savgol_filter(I_raw, 7, 3), 0, None)

    print("\nTEST detect_first_wave sur plusieurs Etats :\n")
    for st in ['New York', 'New Jersey', 'Massachusetts', 'Connecticut',
               'Louisiana', 'California', 'Florida', 'Texas']:
        I = recon(st)
        w = detect_first_wave(I)
        if w:
            print(f"  {st:<16} -> vague J{w['start']}-J{w['end']} "
                  f"(pic J{w['peak']}, I_pic={w['I_peak']:,.0f}) [{w['reason']}]")
        else:
            print(f"  {st:<16} -> AUCUNE vague isolee (candidat Priorite 2)")
