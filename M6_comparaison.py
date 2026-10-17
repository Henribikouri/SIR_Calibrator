"""
=============================================================================
  MODULE M6 -- COMPARAISON DONNEES / SIMULE + TRAJECTOIRE theta(t)
=============================================================================

  Pipeline d'estimation automatique des parametres SIR par IA generative
  Auteur : Henri BIKOURI (Master 2 Genie Logiciel)

  Role de ce module (demonstration centrale de la methodologie) :

    1. Rejouer la calibration reelle (M1 -> M2 -> M3 sequentiel) pour
       obtenir des resultats AUTHENTIQUES (identiques a `python pipeline.py`),
       sur la SERIE COMPLETE et la POPULATION COMPLETE N (plus de fenetrage
       1re vague, plus de pool effectif rho).

    2. EXPORTER deux jeux de donnees lisibles :
       (a) dataset_compare_<prefix>.csv
           -> jour par jour : S/I/R DONNEES vs S/I/R SIMULE (personnes),
              residus et erreur relative, sur les 188 jours complets.
       (b) theta_trajectoire_<prefix>.csv
           -> LA trajectoire theta(t) = {theta_0, theta_1, ..., theta_K} :
              un jeu de parametres (alpha_k, gamma_k, R0_k) par fenetre
              temporelle, re-estime au fur et a mesure de l'evolution de
              l'epidemie. C'est le livrable qui montre EXACTEMENT comment
              theta est genere au cours de la simulation.

    3. PRODUIRE trois figures "memoire-ready" :
       FIG1 : comparaison DONNEES vs SIMULE (I(t) principal + cumul + residus)
       FIG2 : trajectoire theta(t) -- COMMENT theta est genere au fil du temps
              (alpha_k(t), gamma_k(t), R0_k(t) en escalier, alignes sur I(t))
       FIG3 : dynamique DERIVEE de theta(t) dans le temps epidemique
              (R_eff(t) variable, force d'infection, S/I/R sur N complet)

  D'ou part le trace de chaque courbe ?
    - Courbe SIMULEE : concatenation des segments simules fenetre par
      fenetre, avec continuite exacte de l'etat (S,I,R) aux jonctions ->
      voir FIG1 / FIG3.
    - Courbe de THETA(t)  : une valeur par fenetre temporelle (pas par
      generation d'optimiseur) -> voir FIG2, le coeur de la demonstration.

  Utilisation :
    python M6_comparaison.py                 # New York, serie complete (defaut)
    python M6_comparaison.py --state Florida --N 21538187
=============================================================================
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # backend non-interactif (sauvegarde fichier)
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

# Palette coherente et accessible (lisible en clair comme en fonce)
C_DATA   = "#D1495B"   # donnees  (rouge brique)
C_SIM   = "#1B6CA8"   # simule (bleu)
C_S     = "#2E7D32"   # susceptibles (vert)
C_R     = "#8E44AD"   # retires (violet)
C_ACC   = "#E67E22"   # accent (orange) -- prior / reperes
C_GRID  = "#B0B0B0"


# -----------------------------------------------------------------------------
# 1. EXECUTION DE LA CALIBRATION REELLE (M1 -> M2 -> M3 sequentiel)
# -----------------------------------------------------------------------------
def run_calibration(state, N, data_path, disease, mode, D_I):
    """Rejoue la chaine M1->M2->M3 exactement comme pipeline.py (serie
    complete, population complete) et renvoie tous les objets necessaires
    aux exports (meta, prior, m3, calib_result)."""

    # --- M1 : version universelle si dispo, sinon JHU (comme pipeline.py) ---
    try:
        from M1_acquisition_cameroun import AcquisitionModuleUniversel as M1
        use_universal = True
    except ImportError:
        from M1_acquisition import AcquisitionModule as M1
        use_universal = False

    from M2_ia_prior     import IAPriorModule
    from M3_calibration  import CalibrationModule

    print("=" * 65)
    print("  M6 -- COMPARAISON : rejeu de la calibration sequentielle")
    print(f"  Region : {state} | N = {N:,} (population complete) | serie complete")
    print("=" * 65)

    # M1 -- acquisition
    if use_universal:
        m1 = M1(data_path=data_path, N=N, mode=mode, state=state, D_I=D_I)
    else:
        m1 = M1(data_path=data_path, state=state, N=N, D_I=D_I)
    df_sir, meta = m1.run(plot=False)
    meta['disease'] = disease
    meta['state']   = state

    # M2 -- prior IA (fallback litterature si pas d'API : reproductible)
    m2           = IAPriorModule(meta=meta, disease=disease)
    prior_result = m2.run(df_sir=df_sir, plot=False)

    # M3 -- calibration sequentielle (theta_trajectory)
    m3           = CalibrationModule(meta=meta, prior_result=prior_result)
    calib_result = m3.run(df_sir=df_sir, plot=False)

    return meta, prior_result, m3, calib_result


# -----------------------------------------------------------------------------
# 2. RECONSTRUCTION DES SERIES (donnees vs simule, en personnes)
# -----------------------------------------------------------------------------
def build_series(meta, m3):
    """Assemble un DataFrame jour-par-jour : donnees vs simule (personnes)."""
    N   = m3.N
    t   = np.asarray(meta['t_obs'], dtype=float)          # jours
    sol = m3.sol_star                                     # fraction de N (x N = pers.)

    # Donnees (deja en personnes dans meta)
    S_data = np.asarray(meta['S_data'], dtype=float)
    I_data = np.asarray(meta['I_data'], dtype=float)
    R_data = np.asarray(meta['R_data'], dtype=float)

    # Reconstruit (personnes) -- concatenation continue des fenetres theta_k
    S_sim = sol[:, 0] * N
    I_sim = sol[:, 1] * N
    R_sim = sol[:, 2] * N

    peak = float(np.max(I_data)) if np.max(I_data) > 0 else 1.0
    resid_I = I_data - I_sim

    df = pd.DataFrame({
        'jour'          : t.astype(int),
        'S_data'         : S_data,
        'I_data'         : I_data,
        'R_data'         : R_data,
        'S_sim'         : S_sim,
        'I_sim'         : I_sim,
        'R_sim'         : R_sim,
        'residu_I'      : resid_I,
        'err_rel_I_%'   : 100.0 * resid_I / peak,
    })

    for key in ('dates', 'Date', 'date'):
        if key in meta and meta[key] is not None and len(meta[key]) == len(t):
            df.insert(1, 'date', pd.to_datetime(pd.Series(meta[key])).dt.strftime('%Y-%m-%d'))
            break

    return df, peak


def build_dynamics(meta, m3, calib_result):
    """Grandeurs derivees de theta(t), variables dans le TEMPS epidemique.
    Tout est exprime sur la population COMPLETE N (plus de pool effectif) :
    alpha_t/gamma_t sont le regime theta_k EN VIGUEUR a chaque jour t."""
    N    = m3.N
    t    = np.asarray(meta['t_obs'], dtype=float)
    sol  = m3.sol_star                       # fraction de N
    traj = calib_result['theta_trajectory']
    T    = len(t)

    alpha_t = np.zeros(T)
    gamma_t = np.zeros(T)
    for tr in traj:
        alpha_t[tr['day_start']:tr['day_end_step']] = tr['alpha_k']
        gamma_t[tr['day_start']:tr['day_end_step']] = tr['gamma_k']

    s, i, r = sol[:, 0], sol[:, 1], sol[:, 2]   # fraction de N directement

    R_eff = np.divide(alpha_t, gamma_t,
                       out=np.zeros_like(alpha_t), where=gamma_t > 0) * s
    lam   = alpha_t * i                          # force d'infection (/jour/susceptible)
    incid = alpha_t * s * i * N                  # nouvelles infections / jour (personnes)

    dyn = pd.DataFrame({
        'jour'            : t.astype(int),
        'S'               : s * N,
        'I'               : i * N,
        'R_pers'          : r * N,
        'alpha_t'         : alpha_t,
        'gamma_t'         : gamma_t,
        'R_eff(t)'        : R_eff,
        'force_infection' : lam,
        'incidence_/j'    : incid,
    })
    return dyn, R_eff, lam


# -----------------------------------------------------------------------------
# 3. FIGURES
# -----------------------------------------------------------------------------
def _style_ax(ax):
    ax.grid(True, alpha=0.25, color=C_GRID, linewidth=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def fig_comparaison(df, peak, meta, calib_result, save_path):
    """FIG1 -- Dataset DONNEES vs SIMULE (comparaison lisible, serie complete)."""
    t = df['jour'].values
    state = meta.get('state', '')
    rmse = calib_result['RMSE']; r2 = calib_result['R2']; mae = calib_result['MAE']

    fig = plt.figure(figsize=(13, 10))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1.6, 1.2], hspace=0.32)

    # --- (a) I(t) : le coeur de la comparaison ---
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(t, df['I_data'], s=26, color=C_DATA, alpha=0.75,
                zorder=3, label="I(t) DONNEES (dataset)")
    ax1.plot(t, df['I_sim'], color=C_SIM, lw=2.6, zorder=4,
             label="I(t) SIMULE (theta(t), N complet)")
    ax1.scatter([t[0]], [df['I_sim'].iloc[0]], s=90, color=C_SIM,
                edgecolor='white', zorder=6)
    ax1.annotate("depart de la simulation\n(condition initiale i0, t=0)",
                 xy=(t[0], df['I_sim'].iloc[0]),
                 xytext=(t[0] + 0.12 * (t[-1] - t[0]), peak * 0.55),
                 fontsize=9, color=C_SIM,
                 arrowprops=dict(arrowstyle='->', color=C_SIM, lw=1.2))
    kdata = int(np.argmax(df['I_data'].values)); ksim = int(np.argmax(df['I_sim'].values))
    ax1.axvline(t[kdata], color=C_DATA, ls=':', lw=1.2, alpha=0.7)
    ax1.axvline(t[ksim], color=C_SIM, ls=':', lw=1.2, alpha=0.7)
    ax1.set_ylabel("Infectes actifs I(t)  [personnes]")
    ax1.set_title(f"Comparaison DONNEES vs SIMULATION theta(t) -- {state} "
                  f"(serie complete, N entier)",
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)
    txt = (f"R2 = {r2:.4f}\nRMSE = {rmse:,.0f}  ({100*rmse/peak:.1f} % du pic)"
           f"\nMAE  = {mae:,.0f}  ({100*mae/peak:.1f} % du pic)"
           f"\nErreur pic = {abs(t[ksim]-t[kdata])} j")
    ax1.text(0.015, 0.97, txt, transform=ax1.transAxes, va='top', ha='left',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', fc='white', ec=C_GRID, alpha=0.9))
    _style_ax(ax1)

    # --- (b) Cumule infecte : DONNEES vs SIMULE (comparaison honnete) ---
    cum_data = df['I_data'] + df['R_data']          # cas cumules (donnees)
    cum_sim = df['I_sim'] + df['R_sim']          # cas cumules simules (= N - S_sim)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, cum_data, color=C_DATA, ls='--', lw=1.8, label="cumul DONNEES (I+R)")
    ax2.plot(t, cum_sim, color=C_SIM, lw=2.2, label="cumul SIMULE (I+R = N - S)")
    ax2.set_ylabel("Cumule infectes  [pers.]")
    ax2.legend(loc='lower right', ncol=1, fontsize=8, framealpha=0.9)
    ax2.set_title("Sur la population complete, sans pool effectif : theta(t) couvre "
                  "toute la serie, y compris les phases multi-vagues", fontsize=9)
    _style_ax(ax2)

    # --- (c) Residus I_data - I_sim avec bande +/- RMSE ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.axhspan(-rmse, rmse, color=C_SIM, alpha=0.12, label="+/- RMSE")
    ax3.axhline(0, color='black', lw=0.8)
    ax3.bar(t, df['residu_I'], color=C_DATA, alpha=0.65, width=0.9)
    ax3.set_ylabel("Residu\n(donnees - simule)")
    ax3.set_xlabel("Jour")
    ax3.legend(loc='upper right', fontsize=8)
    _style_ax(ax3)

    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return save_path


def fig_theta_trajectoire(theta_trajectory, df, meta, calib_result, save_path):
    """FIG2 -- LA figure centrale : COMMENT theta est genere au fur et a
    mesure de la simulation. Trois panneaux alignes sur le meme axe
    temporel :
      (a) I(t) donnees vs simule, avec une ligne verticale a chaque
          nouvelle fenetre re-estimee ;
      (b) alpha_k(t) et gamma_k(t) en escalier (constants par fenetre) ;
      (c) R0_k(t) en escalier, avec le seuil R0=1 et un code couleur
          (rouge = epidemie en croissance, vert = en declin) -- pour
          visualiser la coincidence entre le franchissement de R0=1 et le
          pic des donnees I(t)."""
    t = df['jour'].values
    state = meta.get('state', '')

    xs     = [tr['day_start'] for tr in theta_trajectory] + \
             [theta_trajectory[-1]['day_end_step']]
    alphas = [tr['alpha_k'] for tr in theta_trajectory]
    gammas = [tr['gamma_k'] for tr in theta_trajectory]
    r0s    = [tr['R0_k']    for tr in theta_trajectory]

    fig = plt.figure(figsize=(14, 12))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 1], hspace=0.34)

    # --- (a) I(t) donnees vs simule + frontieres de fenetre ---
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(t, df['I_data'], s=22, color=C_DATA, alpha=0.7, zorder=3,
                label="I(t) DONNEES")
    ax1.plot(t, df['I_sim'], color=C_SIM, lw=2.4, zorder=4,
             label="I(t) SIMULE")
    for xb in xs:
        ax1.axvline(xb, color=C_GRID, ls=':', lw=0.9, alpha=0.7)
    ax1.set_ylabel("Infectes actifs I(t)")
    ax1.set_title(f"Comment theta(t) est genere au fil de la simulation -- {state}\n"
                  f"chaque ligne verticale = une nouvelle fenetre theta_k re-estimee "
                  f"({len(theta_trajectory)} fenetres)",
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    _style_ax(ax1)

    # --- (b) alpha_k(t) / gamma_k(t) en escalier ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.step(xs, alphas + [alphas[-1]], where='post',
             color='darkorange', lw=2.2, label="alpha_k (transmission)")
    ax2.step(xs, gammas + [gammas[-1]], where='post',
             color='teal', lw=2.2, label="gamma_k (guerison)")
    for xb in xs:
        ax2.axvline(xb, color=C_GRID, ls=':', lw=0.9, alpha=0.7)
    ax2.set_ylabel("Valeur du parametre")
    ax2.set_title("theta_k = (alpha_k, gamma_k) -- un regime par fenetre, chaine par "
                  "continuite exacte de l'etat (S,I,R)", fontsize=10.5)
    ax2.legend(loc='best', fontsize=9)
    _style_ax(ax2)

    # --- (c) R0_k(t) en escalier, seuil = 1 ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.step(xs, r0s + [r0s[-1]], where='post', color='#333333', lw=2.4, zorder=4)
    ax3.axhline(1.0, color=C_DATA, ls='--', lw=1.4, zorder=3)
    for k in range(len(xs) - 1):
        color = '#F5B7B1' if r0s[k] > 1 else '#A9DFBF'
        ax3.axvspan(xs[k], xs[k + 1], color=color, alpha=0.35, zorder=1)
    ax3.set_ylabel("R0_k = alpha_k / gamma_k")
    ax3.set_xlabel("Jour")
    ax3.set_title("R0_k franchit 1 : rouge = l'epidemie croit, vert = elle recule "
                  "-- a comparer au pic de I(t) ci-dessus (panneau a)", fontsize=10.5)
    # Plafonne l'axe pour la lisibilite : les tout premiers regimes (signal
    # quasi nul, cf. M3) peuvent produire un R0_k ponctuellement tres eleve
    # sans consequence sur la reconstruction (I(t) y est negligeable) --
    # on l'indique plutot que de laisser un seul point ecraser tout l'axe.
    r0_cap = max(4.0, float(np.percentile(r0s, 90)) * 1.3)
    if max(r0s) > r0_cap:
        ax3.set_ylim(0, r0_cap)
        ax3.text(0.99, 0.96, f"axe plafonne a {r0_cap:.1f} (max reel = {max(r0s):.1f}, "
                  f"phase a signal quasi nul, cf. FIG2 panneau b)",
                  transform=ax3.transAxes, ha='right', va='top', fontsize=7.5,
                  style='italic', color='dimgray')
    _style_ax(ax3)

    fig.suptitle(f"Trajectoire theta(t) -- {state} (population complete N, "
                 f"pas de pool effectif)", fontsize=14, fontweight='bold', y=1.0)
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return save_path


def fig_dynamique(dyn, R_eff, lam, meta, calib_result, save_path):
    """FIG3 -- Dynamique derivee de theta(t) dans le temps epidemique, sur
    la population COMPLETE N (pas de pool effectif)."""
    t = dyn['jour'].values
    state = meta.get('state', '')

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.subplots_adjust(hspace=0.28)

    # --- R_eff(t) (variable : depend du regime theta_k courant) ---
    ax = axes[0]
    ax.plot(t, R_eff, color=C_SIM, lw=2.6)
    ax.axhline(1.0, color=C_DATA, ls='--', lw=1.5)
    ax.text(t[-1], 1.02, "R_eff = 1  (seuil : au-dessus l'epidemie croit, en-dessous elle recule)",
            ha='right', va='bottom', fontsize=9, color=C_DATA)
    crosses = np.where(np.diff(np.sign(R_eff - 1.0)))[0]
    for kc in crosses:
        ax.axvline(t[kc], color='gray', ls=':', lw=1.0, alpha=0.6)
    if len(crosses):
        kc = crosses[0]
        ax.annotate(f"R_eff franchit 1\n(jour ~{int(t[kc])})", xy=(t[kc], 1.0),
                    xytext=(10, 20), textcoords='offset points', fontsize=9, color='dimgray',
                    arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_ylabel("R_eff(t) = (alpha_t/gamma_t)*S(t)/N")
    ax.set_title(f"Dynamique derivee de theta(t) dans le temps -- {state} "
                 f"(population COMPLETE N)", fontsize=14, fontweight='bold')
    _style_ax(ax)

    # --- Force d'infection ---
    ax = axes[1]
    ax.plot(t, lam, color=C_ACC, lw=2.4)
    ax.fill_between(t, 0, lam, color=C_ACC, alpha=0.15)
    ax.set_ylabel("Force d'infection\nlambda(t) = alpha_t*I(t)/N  [/jour]")
    ax.set_title("Pression de transmission subie par un susceptible -- varie avec "
                 "le regime theta_k en vigueur", fontsize=10)
    _style_ax(ax)

    # --- S / I / R sur la population complete ---
    ax = axes[2]
    ax.plot(t, dyn['S'], color=C_S, lw=2.2, label="S(t) susceptibles")
    ax.plot(t, dyn['I'], color=C_DATA, lw=2.6, label="I(t) infectes actifs")
    ax.plot(t, dyn['R_pers'], color=C_R, lw=2.2, label="R(t) retires")
    ax.set_ylabel("Effectifs sur N\n[personnes]")
    ax.set_xlabel("Jour")
    ax.set_title("Ce n'est PAS la depletion de S(t) qui freine I(t) (S(t) reste proche "
                 "de N) : c'est la chute de R_eff(t) au changement de regime theta_k",
                 fontsize=9.5)
    ax.legend(loc='center right', framealpha=0.9)
    _style_ax(ax)

    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return save_path


# -----------------------------------------------------------------------------
# 4. ORCHESTRATION
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="M6 -- comparaison donnees/simule + trajectoire theta(t)")
    ap.add_argument('--data',    default="data/usa_county_wise.csv")
    ap.add_argument('--state',   default="New York")
    ap.add_argument('--N',       type=int, default=19_450_000)
    ap.add_argument('--disease', default="COVID-19")
    ap.add_argument('--mode',    default="auto")
    ap.add_argument('--D-I',     type=int, default=14, dest='D_I')
    ap.add_argument('--outdir',  default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prefix = f"{args.state.replace(' ', '_').replace(chr(39), '')}_{args.N}"

    # 1. Calibration reelle
    meta, prior_result, m3, calib_result = run_calibration(
        args.state, args.N, args.data, args.disease, args.mode, args.D_I)

    # 2. Datasets
    df, peak        = build_series(meta, m3)
    dyn, R_eff, lam  = build_dynamics(meta, m3, calib_result)
    theta_trajectory = calib_result['theta_trajectory']

    csv_cmp = os.path.join(args.outdir, f"dataset_compare_{prefix}.csv")
    csv_thj = os.path.join(args.outdir, f"theta_trajectoire_{prefix}.csv")
    csv_dyn = os.path.join(args.outdir, f"dynamique_temporelle_{prefix}.csv")
    df.to_csv(csv_cmp, index=False, float_format="%.3f")
    pd.DataFrame(theta_trajectory).to_csv(csv_thj, index=False, float_format="%.8f")
    dyn.to_csv(csv_dyn, index=False, float_format="%.4f")

    # 3. Figures
    f1 = fig_comparaison(df, peak, meta, calib_result,
                         os.path.join(args.outdir, f"FIG1_comparaison_{prefix}.png"))
    f2 = fig_theta_trajectoire(theta_trajectory, df, meta, calib_result,
                         os.path.join(args.outdir, f"FIG2_trajectoire_theta_{prefix}.png"))
    f3 = fig_dynamique(dyn, R_eff, lam, meta, calib_result,
                       os.path.join(args.outdir, f"FIG3_dynamique_{prefix}.png"))

    # 4. Synthese console
    print("\n" + "=" * 65)
    print("  M6 -- FICHIERS GENERES")
    print("=" * 65)
    print(f"  Datasets :")
    print(f"   -> {csv_cmp}")
    print(f"   -> {csv_thj}   ({len(theta_trajectory)} fenetres theta_k)")
    print(f"   -> {csv_dyn}")
    print(f"  Figures  :")
    print(f"   -> {f1}")
    print(f"   -> {f2}")
    print(f"   -> {f3}")
    print(f"\n  Dernier regime : alpha={calib_result['alpha_star']:.4f}  "
          f"gamma={calib_result['gamma_star']:.4f}  "
          f"R0={calib_result['R0_star']:.3f}")
    print(f"  R2={calib_result['R2']:.4f}  RMSE={calib_result['RMSE']:,.0f}  "
          f"({100*calib_result['RMSE']/peak:.1f}% du pic)  "
          f"-- reconstruction sur {len(df)} jours, N={m3.N:,} (population complete)")
    print("=" * 65)


if __name__ == "__main__":
    main()
