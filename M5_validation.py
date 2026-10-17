"""
=============================================================================
  MODULE M5 -- VALIDATION PROSPECTIVE ET ANALYSE DE SENSIBILITE
=============================================================================

  Pipeline d'estimation automatique des parametres SIR (alpha, beta', gamma)
  par Intelligence Artificielle generative

  Auteur  : Henri (Master 2 Genie Logiciel)

  Role de ce module :
    - Valider la qualite de la reconstruction theta(t) par une validation
      PROSPECTIVE one-step-ahead (au lieu d'un split train/test fixe)
    - Analyser la sensibilite du modele SIR aux parametres actifs (Sobol)
    - Generer le rapport final complet de validation
    - Produire toutes les figures pour le memoire

  Contenu scientifique :
    1. Validation prospective one-step-ahead
       Pour chaque fenetre k>=1 de la trajectoire theta(t) calculee par M3,
       on simule la fenetre k EN N'UTILISANT QUE le regime theta_{k-1}
       (calibre AVANT d'avoir vu les donnees de la fenetre k) applique a
       l'etat reellement chaine (S,I,R) au debut de la fenetre k. C'est une
       validation strictement causale : theta_{k-1} n'a jamais vu I_data sur
       la fenetre k. Contrairement a un split fixe 70/30, elle produit un
       backtest sur ~N_fenetres-1 points plutot qu'un seul, et s'inscrit
       naturellement dans la logique "au fur et a mesure" de la refonte
       sequentielle (cf. METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md).

    2. Analyse de sensibilite de Sobol
       Quantifie l'importance relative de alpha et gamma (les 2 SEULS
       parametres actifs -- beta' est structurellement absent de l'EDO,
       et il n'y a plus de rho depuis la refonte). Appliquee sur la fenetre
       theta_k qui contient le pic epidemique observe -- la fenetre la plus
       informative, ou la dynamique bascule (R0_k proche de 1).
       Indices S1 (effet direct) et ST (effet total).

    3. Rapport final complet
       Synthese de tous les resultats du pipeline M1->M5, y compris la
       trajectoire theta(t) complete (tracabilite scientifique).

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.integrate import odeint
from SALib.sample import saltelli
from SALib.analyze import sobol
import os
import json
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# CLASSE PRINCIPALE : ValidationModule
# -----------------------------------------------------------------------------
class ValidationModule:
    """
    Module M5 du pipeline de calibration SIR -- version sequentielle.

    Effectue la validation complete de la reconstruction theta(t) :
    - Validation prospective one-step-ahead (remplace la validation croisee
      a split fixe, devenue caduque puisqu'il n'y a plus un theta* unique)
    - Analyse de sensibilite de Sobol (alpha, gamma)
    - Rapport final de synthese

    Attributs
    ---------
    meta             : dict -- metadonnees de M1
    prior_result     : dict -- resultats de M2
    calib_result     : dict -- resultats de M3 (theta_trajectory, sol_star...)
    N                : int  -- taille de la population COMPLETE
    theta_trajectory : list[dict] -- trajectoire theta_k issue de M3
    """

    def __init__(self, meta, prior_result, calib_result):
        """
        Initialise M5 avec les sorties de M1, M2 et M3.

        Parameters
        ----------
        meta         : dict -- sortie de M1.run()
        prior_result : dict -- sortie de M2.run()
        calib_result : dict -- sortie de M3.run() (theta_trajectory, sol_star...)
        """
        self.meta         = meta
        self.prior_result = prior_result
        self.calib_result = calib_result
        self.N            = meta['N']

        # Donnees normalisees [0,1] pour coherence avec M3
        self.t_obs = meta['t_obs'].astype(float)
        self.S_data = meta['S_data'] / self.N
        self.I_data = meta['I_data'] / self.N
        self.R_data = meta['R_data'] / self.N
        self.T     = len(self.t_obs)
        self.peak_data = float(np.max(self.I_data)) * self.N   # pic observe (personnes)

        # Dernier regime connu (compat / affichage), et TRAJECTOIRE complete
        self.alpha_star      = calib_result['alpha_star']
        self.beta_prime_star = calib_result['beta_prime_star']
        self.gamma_star      = calib_result['gamma_star']
        self.R0_star         = calib_result['R0_star']
        self.theta_trajectory = calib_result['theta_trajectory']
        self.n_windows         = calib_result['n_windows']

        # Resultats de validation -- remplis apres execution
        self.prospective_result = None
        self.sobol_result       = None

        print("=" * 65)
        print("  MODULE M5 -- Validation Prospective et Analyse de Sensibilite")
        print("=" * 65)
        print(f"  Trajectoire theta(t) : {self.n_windows} fenetres")
        print(f"  Dernier regime :")
        print(f"  -> alpha* = {self.alpha_star:.6f}")
        print(f"  -> beta'* = {self.beta_prime_star} jours")
        print(f"  -> gamma* = {self.gamma_star:.6f}")
        print(f"  -> R0*    = {self.R0_star:.4f}")
        print(f"  Serie     : {self.T} jours")
        print("=" * 65)

    # -------------------------------------------------------------------------
    # SOLVEUR SIR interne
    # -------------------------------------------------------------------------
    def _sir_ode(self, y, t, alpha, gamma):
        """
        Systeme SIR normalise [0,1] sur la population COMPLETE.
        Identique a M3 pour coherence des resultats.
        """
        S, I, R = y
        dS = -alpha * S * I
        dI =  alpha * S * I - gamma * I
        dR =  gamma * I
        return [dS, dI, dR]

    def _solve_sir(self, alpha, gamma, S0, I0, R0, t_array):
        """
        Resout le systeme SIR sur la population COMPLETE, a partir d'une
        condition initiale (S0, I0, R0) donnee (fractions de N, somme=1).
        Coherent avec M3 : pas de pool effectif, sol*N = personnes.
        """
        try:
            sol = odeint(
                self._sir_ode, [S0, I0, R0], t_array,
                args=(alpha, gamma),
                rtol=1e-7, atol=1e-9, mxstep=5000
            )
            return sol
        except Exception:
            return None

    def _rmse(self, I_data, I_sim):
        """Calcule le RMSE en cas/jour (echelle N)."""
        return float(np.sqrt(np.mean((I_data - I_sim) ** 2)) * self.N)

    def _r2(self, I_data, I_sim):
        """Calcule le R2."""
        ss_res = np.sum((I_data - I_sim) ** 2)
        ss_tot = np.sum((I_data - np.mean(I_data)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # -------------------------------------------------------------------------
    # ETAPE 1 : Validation prospective one-step-ahead
    # -------------------------------------------------------------------------
    def run_prospective_validation(self):
        """
        Validation prospective one-step-ahead sur la trajectoire theta(t).

        Principe :
        Pour chaque fenetre k (k=1..K-1) de la trajectoire theta(t) issue de
        M3, on simule la fenetre k en appliquant le regime theta_{k-1} --
        calibre AVANT d'avoir vu les donnees de la fenetre k -- a partir de
        l'etat (S,I,R) reellement chaine au debut de la fenetre k. On compare
        cette prevision a I_data sur la fenetre k.

        Pourquoi cette validation ?
        C'est un test de generalisation strictement causal (jamais de fuite
        d'information du futur vers le passe), et elle exploite directement
        la structure sequentielle de la calibration : c'est la meilleure
        facon de repondre a "le theta qu'on vient d'apprendre aurait-il
        predit correctement la suite de l'epidemie ?". Contrairement a
        l'ancien split fixe 70/30, elle fournit ~K-1 points de backtest
        (un par fenetre), donc une estimation bien plus robuste.

        Returns
        -------
        self -- chaine les methodes
        """
        print(f"\n[M5 - Etape 1] Validation prospective one-step-ahead...")

        traj = self.theta_trajectory
        if len(traj) < 2:
            print("  !! Pas assez de fenetres pour une validation prospective "
                  "(1 seule fenetre theta(t)) -- etape ignoree.")
            self.prospective_result = None
            return self

        per_window = []
        all_data, all_sim = [], []

        for k in range(1, len(traj)):
            prev = traj[k - 1]
            cur  = traj[k]
            day_start, day_end = cur['day_start'], cur['day_end_step']

            t_slice = self.t_obs[day_start:day_end]
            # Etat de depart REEL (chaine par M3), regime = celui appris AVANT
            # d'avoir vu cette fenetre (prev) -- test causal.
            S0, I0, R0 = cur['S_start'], cur['I_start'], cur['R_start']

            sol = self._solve_sir(prev['alpha_k'], prev['gamma_k'], S0, I0, R0, t_slice)
            if sol is None:
                continue

            I_data_slice = self.I_data[day_start:day_end]
            I_sim_slice = sol[:, 1]

            per_window.append({
                'k'         : k,
                'day_start' : int(day_start),
                'day_end'   : int(day_end),
                'rmse'      : self._rmse(I_data_slice, I_sim_slice),
                'r2'        : self._r2(I_data_slice, I_sim_slice),
                'I_data'     : I_data_slice,
                'I_sim'     : I_sim_slice,
            })
            all_data.append(I_data_slice)
            all_sim.append(I_sim_slice)

        if not all_data:
            print("  !! Aucune prevision valide -- validation prospective annulee")
            self.prospective_result = None
            return self

        I_data_cat = np.concatenate(all_data)
        I_sim_cat = np.concatenate(all_sim)
        rmse_global = self._rmse(I_data_cat, I_sim_cat)
        r2_global   = self._r2(I_data_cat, I_sim_cat)

        self.prospective_result = {
            'n_forecasts'      : len(per_window),
            'rmse_prospectif'  : round(rmse_global, 2),
            'r2_prospectif'    : round(r2_global,   4),
            'per_window'       : per_window,
        }

        print(f"\n  RESULTATS VALIDATION PROSPECTIVE ({len(per_window)} previsions "
              f"one-step-ahead) :")
        print(f"  {'Metrique':<25} {'Valeur'}")
        print(f"  {'-'*45}")
        print(f"  {'RMSE prospectif (cas/j)':<25} {rmse_global:.1f}")
        print(f"  {'R2 prospectif':<25} {r2_global:.4f}")
        print(f"  {'-'*45}")
        if r2_global > 0.80:
            interp = "BON -- la trajectoire theta(t) generalise bien"
        elif r2_global > 0.50:
            interp = "ACCEPTABLE -- generalisation partielle"
        else:
            interp = "ATTENTION -- faible capacite predictive one-step-ahead"
        print(f"  Interpretation : {interp}")

        return self

    # -------------------------------------------------------------------------
    # ETAPE 2 : Analyse de sensibilite de Sobol (sur la fenetre du pic)
    # -------------------------------------------------------------------------
    def run_sobol_analysis(self, n_samples=512):
        """
        Effectue l'analyse de sensibilite globale de Sobol sur (alpha, gamma)
        -- les 2 SEULS parametres actifs de theta(t) (beta' est
        structurellement absent de l'EDO continue, et il n'y a plus de rho).

        Appliquee sur la fenetre theta_k qui CONTIENT le pic epidemique
        observe : c'est la fenetre la plus informative scientifiquement
        (bascule de R0_k autour de 1), et cela evite un budget Sobol
        demultiplie par le nombre de fenetres (~25).

        Parameters
        ----------
        n_samples : int -- nombre d'echantillons Saltelli (multiple de 2)

        Returns
        -------
        self -- chaine les methodes
        """
        print(f"\n[M5 - Etape 2] Analyse de sensibilite de Sobol...")

        peak_idx = int(np.argmax(self.I_data))
        win_idx = len(self.theta_trajectory) - 1
        for i, tr in enumerate(self.theta_trajectory):
            if tr['day_start'] <= peak_idx < tr['day_end_step']:
                win_idx = i
                break
        tr = self.theta_trajectory[win_idx]

        S0, I0, R0 = tr['S_start'], tr['I_start'], tr['R_start']
        t_slice = self.t_obs[tr['day_start']:tr['day_end_step']]

        print(f"  -> Fenetre analysee : {win_idx} (jours {tr['day_start']}-"
              f"{tr['day_end_step']-1}, contient le pic observe J{peak_idx})")
        print(f"  -> N echantillons : {n_samples} x (2*2+2) = {n_samples * 6} evaluations SIR")
        print(f"  -> Sortie analysee : pic local max(I(t)) sur cette fenetre")

        problem = {
            'num_vars' : 2,
            'names'    : ['alpha', 'gamma'],
            'bounds'   : [
                list(self.prior_result['bounds'][0]),  # alpha
                list(self.prior_result['bounds'][2]),  # gamma
            ]
        }

        param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
        print(f"  -> {len(param_values)} evaluations du modele SIR...")

        Y = np.zeros(len(param_values))
        for i, params in enumerate(param_values):
            alpha, gamma = params[0], params[1]
            sol = self._solve_sir(alpha, gamma, S0, I0, R0, t_slice)
            Y[i] = float(np.max(sol[:, 1])) if sol is not None else 0.0

        Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

        self.sobol_result = {
            'S1'      : {'alpha': float(Si['S1'][0]), 'gamma': float(Si['S1'][1])},
            'ST'      : {'alpha': float(Si['ST'][0]), 'gamma': float(Si['ST'][1])},
            'S1_conf' : {'alpha': float(Si['S1_conf'][0]), 'gamma': float(Si['S1_conf'][1])},
            'ST_conf' : {'alpha': float(Si['ST_conf'][0]), 'gamma': float(Si['ST_conf'][1])},
            'n_samples'     : n_samples,
            'n_eval'        : len(param_values),
            'window_index'  : win_idx,
            'window_days'   : [tr['day_start'], tr['day_end_step']],
        }

        print(f"\n  INDICES DE SOBOL (fenetre {win_idx}) :")
        print(f"  {'Parametre':<15} {'S1 (direct)':<18} {'ST (total)'}")
        print(f"  {'-'*50}")
        for p in ['alpha', 'gamma']:
            s1, st = self.sobol_result['S1'][p], self.sobol_result['ST'][p]
            s1c, stc = self.sobol_result['S1_conf'][p], self.sobol_result['ST_conf'][p]
            print(f"  {p:<15} {s1:.4f} +/- {s1c:.4f}    {st:.4f} +/- {stc:.4f}")

        s1_vals = self.sobol_result['S1']
        dominant = max(s1_vals, key=s1_vals.get)
        print(f"\n  -> Parametre dominant : {dominant} (S1={s1_vals[dominant]:.4f})")
        print(f"  -> Interpretation : {dominant} explique "
              f"{s1_vals[dominant]*100:.1f}% de la variance du pic local")

        return self

    # -------------------------------------------------------------------------
    # ETAPE 3 : Graphiques complets de validation
    # -------------------------------------------------------------------------
    def plot_validation(self, df_sir,
                         save_path="outputs/M5_validation.png"):
        """
        Genere le graphique complet de validation en 4 panneaux :

        Panneau 1 : Validation prospective one-step-ahead
          Chaque segment colore = prevision faite par le regime PRECEDENT,
          jamais entraine sur les donnees affichees dans ce segment.

        Panneau 2 : Indices de Sobol S1 et ST (alpha, gamma)

        Panneau 3 : Reconstruction complete theta(t) vs observe
          Vue d'ensemble sur les 188 jours (population complete).

        Panneau 4 : Tableau de synthese des resultats
        """
        os.makedirs("outputs", exist_ok=True)

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f"Module M5 -- Validation Prospective | "
            f"{self.calib_result.get('disease', 'COVID-19')} | {self.calib_result.get('state', 'Unknown')}\n"
            f"{self.n_windows} fenetres theta(t) | dernier regime : "
            f"alpha*={self.alpha_star:.4f}  "
            f"beta'*={self.beta_prime_star}j  "
            f"gamma*={self.gamma_star:.4f}  "
            f"R0*={self.R0_star:.3f}",
            fontsize=13, fontweight='bold'
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # -- Panneau 1 : Validation prospective one-step-ahead ----
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.t_obs, self.I_data * self.N,
                 color='black', linewidth=1.2, alpha=0.35,
                 label='I(t) donnees (reference)')
        pr = self.prospective_result
        if pr:
            for i, w in enumerate(pr['per_window']):
                t_seg = self.t_obs[w['day_start']:w['day_end']]
                ax1.plot(t_seg, w['I_sim'] * self.N,
                         color='blue', linewidth=2,
                         label='Prevision one-step-ahead (theta_{k-1})' if i == 0 else None)
                ax1.axvline(t_seg[0], color='gray', linestyle=':', alpha=0.4)
            ax1.set_title(f"Validation prospective one-step-ahead\n"
                          f"R2={pr['r2_prospectif']:.3f}  "
                          f"RMSE={pr['rmse_prospectif']:,.0f}  "
                          f"({pr['n_forecasts']} previsions)",
                          fontsize=11)
        else:
            ax1.set_title("Validation prospective non disponible\n"
                          "(1 seule fenetre theta(t))", fontsize=11)
        ax1.set_ylabel('Infectes actifs I(t)')
        ax1.set_xlabel('Jours')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # -- Panneau 2 : Indices de Sobol (alpha, gamma) -----------
        ax2 = fig.add_subplot(gs[0, 1])
        if self.sobol_result:
            sr = self.sobol_result
            params_labels = ['alpha\n(infection)', 'gamma\n(guerison)']
            s1_vals = [sr['S1']['alpha'], sr['S1']['gamma']]
            st_vals = [sr['ST']['alpha'], sr['ST']['gamma']]
            s1_conf = [sr['S1_conf']['alpha'], sr['S1_conf']['gamma']]
            st_conf = [sr['ST_conf']['alpha'], sr['ST_conf']['gamma']]

            x, width = np.arange(len(params_labels)), 0.35
            bars1 = ax2.bar(x - width/2, s1_vals, width,
                             color=['#E74C3C', '#2ECC71'],
                             alpha=0.8, label='S1 (effet direct)')
            ax2.bar(x + width/2, st_vals, width,
                    color=['#C0392B', '#27AE60'],
                    alpha=0.5, label='ST (effet total)', hatch='//')

            ax2.errorbar(x - width/2, s1_vals, yerr=s1_conf, fmt='none',
                         color='black', capsize=5, linewidth=1.5)
            ax2.errorbar(x + width/2, st_vals, yerr=st_conf, fmt='none',
                         color='black', capsize=5, linewidth=1.5)

            for bar, val in zip(bars1, s1_vals):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                         fontweight='bold')

            wd = sr['window_days']
            ax2.set_title(f"Sensibilite de Sobol -- fenetre du pic\n"
                          f"(jours {wd[0]}-{wd[1]-1})", fontsize=11)
            ax2.set_xticks(x)
            ax2.set_xticklabels(params_labels, fontsize=10)
            ax2.set_ylabel('Indice de Sobol')
            ax2.set_ylim(0, 1.1)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')

        # -- Panneau 3 : Reconstruction complete theta(t) ----------
        ax3 = fig.add_subplot(gs[1, 0])
        sol_full = self.calib_result.get('sol_star')
        if sol_full is not None:
            t = self.t_obs
            ax3.plot(t, self.I_data * self.N, color='red', linewidth=2.5,
                     label='I(t) donnees')
            ax3.plot(t, sol_full[:, 1] * self.N, color='blue', linewidth=2,
                     linestyle='--',
                     label=f'I(t) simule theta(t) (R2={self.calib_result["R2"]:.3f})')
            ax3.fill_between(t, self.I_data * self.N, sol_full[:, 1] * self.N,
                             alpha=0.15, color='purple', label='Ecart residuel')
            peak_data = int(np.argmax(self.I_data))
            peak_sim = int(np.argmax(sol_full[:, 1]))
            ax3.axvline(t[peak_data], color='red', linestyle=':', alpha=0.7,
                        label=f'Pic donnees J{peak_data}')
            ax3.axvline(t[peak_sim], color='blue', linestyle=':', alpha=0.7,
                        label=f'Pic simule J{peak_sim}')

        ax3.set_title('Simulation complete theta(t) -- Vue d\'ensemble\n'
                      'Donnees (plein) vs Simule (pointille), N entier',
                      fontsize=11)
        ax3.set_ylabel('Infectes actifs I(t)')
        ax3.set_xlabel('Jours')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # -- Panneau 4 : Tableau de synthese ----------------------
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        cr = self.calib_result
        pr = self.prospective_result or {}
        sr = self.sobol_result or {}

        rows = [
            ['DERNIER REGIME theta_K', '', ''],
            ['alpha* (infection_rate)',
             f"{self.alpha_star:.6f}", f"prior: {cr['alpha_prior']:.4f}"],
            ["beta'* (min_inf_time)",
             f"{self.beta_prime_star} j", f"prior: {cr['beta_prime_prior']} j"],
            ['gamma* (recovery_rate)',
             f"{self.gamma_star:.6f}", f"prior: {cr['gamma_prior']:.4f}"],
            ['R0* = alpha*/gamma*',
             f"{self.R0_star:.4f}", f"prior: {cr['R0_prior']:.3f}"],
            ['Nb. fenetres theta_k', f"{self.n_windows}", ''],
            ['Duree moy. maladie', f"{cr['duree_maladie']:.1f} j", ''],
            ['', '', ''],
            ['METRIQUES RECONSTRUCTION (globale)', '', ''],
            ['RMSE (% pic)',
             f"{100*cr['RMSE']/max(self.peak_data,1):.1f}%", '< 5%'],
            ['R2', f"{cr['R2']:.4f}", '> 0.95'],
            ['Erreur pic (j)', f"{cr['peak_day_err']}", '< 7'],
            ['', '', ''],
            ['VALIDATION PROSPECTIVE (one-step-ahead)', '', ''],
            ['RMSE prospectif',
             f"{pr.get('rmse_prospectif', 'N/A')}", ''],
            ['R2 prospectif',
             f"{pr.get('r2_prospectif', 'N/A')}", '> 0.80'],
            ['Nb. previsions',
             f"{pr.get('n_forecasts', 'N/A')}", ''],
            ['', '', ''],
            ['SENSIBILITE SOBOL (S1, fenetre du pic)', '', ''],
            ['alpha (dominant ?)',
             f"{sr['S1']['alpha']:.4f}" if sr else 'N/A', ''],
            ['gamma',
             f"{sr['S1']['gamma']:.4f}" if sr else 'N/A', ''],
        ]

        table = ax4.table(
            cellText  = [[r[0], r[1], r[2]] for r in rows],
            colLabels = ['Element', 'Valeur', 'Reference'],
            loc       = 'center',
            cellLoc   = 'left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.35)

        for j in range(3):
            table[0, j].set_facecolor('#1F4E79')
            table[0, j].set_text_props(color='white', fontweight='bold')

        for i, row in enumerate(rows, 1):
            if row[0] in ['DERNIER REGIME theta_K', 'METRIQUES RECONSTRUCTION (globale)',
                          'VALIDATION PROSPECTIVE (one-step-ahead)',
                          'SENSIBILITE SOBOL (S1, fenetre du pic)']:
                for j in range(3):
                    table[i, j].set_facecolor('#D6E4F0')
                    table[i, j].set_text_props(fontweight='bold')

        ax4.set_title('Synthese complete du pipeline M1-M5',
                      fontsize=11, pad=18, y=1.04)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  -> Graphique sauvegarde : {save_path}")

    # -------------------------------------------------------------------------
    # ETAPE 4 : Rapport final JSON
    # -------------------------------------------------------------------------
    def generate_final_report(self):
        """
        Genere le rapport final JSON de synthese du pipeline complet.

        Ce rapport contient tous les resultats de M1 a M5, y compris la
        trajectoire theta(t) complete, et constitue la trace scientifique
        complete de la calibration sequentielle.
        """
        print(f"\n[M5 - Etape 4] Generation du rapport final...")

        def safe(v):
            if isinstance(v, (np.integer,)):  return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, np.ndarray):     return v.tolist()
            return v

        def clean(d):
            if isinstance(d, dict):
                return {k: clean(v) for k, v in d.items()
                        if not isinstance(v, np.ndarray)}
            elif isinstance(d, list):
                return [clean(i) for i in d]
            else:
                return safe(d)

        cr = self.calib_result
        pr = self.prospective_result or {}
        sr = self.sobol_result or {}

        report = {
            "pipeline"    : "SIR Calibration Pipeline v2.0 -- calibration sequentielle theta(t)",
            "auteur"      : "Henri -- Master 2 Genie Logiciel",
            "disease"     : cr.get('disease', 'COVID-19'),
            "region"      : cr.get('state',   'New York'),
            "modules"     : {
                "M1" : "Acquisition et pretraitement",
                "M2" : "Extraction prior par IA generative (Gemini/Claude)",
                "M3" : "Calibration sequentielle theta(t) (fenetres glissantes DE+Nelder-Mead)",
                "M4" : "Generation calendrier GAML piecewise",
                "M5" : "Validation prospective et analyse de sensibilite"
            },
            "dernier_regime" : {
                "alpha_star"      : safe(self.alpha_star),
                "beta_prime_star" : safe(self.beta_prime_star),
                "gamma_star"      : safe(self.gamma_star),
                "R0_star"         : safe(self.R0_star),
                "duree_maladie"   : safe(cr.get('duree_maladie', 0))
            },
            "theta_trajectory" : clean(cr.get('theta_trajectory', [])),
            "n_windows"        : safe(cr.get('n_windows', 0)),
            "metriques_reconstruction" : {
                "RMSE" : safe(cr['RMSE']),
                "MAE"  : safe(cr['MAE']),
                "R2"   : safe(cr['R2']),
                "peak_day_err" : safe(cr['peak_day_err'])
            },
            "validation_prospective" : {
                "n_forecasts"     : safe(pr.get('n_forecasts', 0)),
                "rmse_prospectif" : safe(pr.get('rmse_prospectif', 0)),
                "r2_prospectif"   : safe(pr.get('r2_prospectif', 0)),
            },
            "sensibilite_sobol" : {
                "S1"           : clean(sr.get('S1', {})),
                "ST"           : clean(sr.get('ST', {})),
                "window_index" : safe(sr.get('window_index', -1)),
                "window_days"  : safe(sr.get('window_days', [])),
                "n_samples"    : safe(sr.get('n_samples', 0)),
                "n_eval"       : safe(sr.get('n_eval',    0))
            },
            "injection_gama" : {
                "note" : "calendrier piecewise complet dans params_sir_calibrated.gaml "
                         "(voir regime_calendar) -- ces valeurs sont le DERNIER regime",
                "infection_rate"      : safe(self.alpha_star),
                "min_infectious_time" : safe(self.beta_prime_star),
                "recovery_rate"       : safe(self.gamma_star),
                "R0_calibrated"       : safe(self.R0_star)
            }
        }

        os.makedirs("outputs", exist_ok=True)
        path = "outputs/rapport_final_pipeline.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"  -> Rapport final : {path}")
        return path

    # -------------------------------------------------------------------------
    # METHODE PRINCIPALE : run()
    # -------------------------------------------------------------------------
    def run(self, df_sir, plot=True, n_sobol=512, save_path="outputs/M5_validation.png"):
        """
        Orchestre toutes les etapes du Module M5.

        Etapes :
        1. Validation prospective one-step-ahead sur theta(t)
        2. Analyse de sensibilite de Sobol (fenetre du pic)
        3. Graphiques complets de validation
        4. Rapport final JSON

        Parameters
        ----------
        df_sir   : pd.DataFrame -- donnees SIR de M1
        plot     : bool         -- generer les graphiques
        n_sobol  : int          -- echantillons Sobol (512 recommande)

        Returns
        -------
        validation_result : dict -- resultats complets de validation
        """
        self.run_prospective_validation()
        self.run_sobol_analysis(n_samples=n_sobol)

        if plot:
            self.plot_validation(df_sir, save_path=save_path)

        report_path = self.generate_final_report()

        validation_result = {
            'prospective_result' : self.prospective_result,
            'sobol_result'       : self.sobol_result,
            'report_path'        : report_path
        }

        print("\n" + "=" * 65)
        print("  MODULE M5 -- TERMINE AVEC SUCCES")
        print("=" * 65)
        if self.prospective_result:
            pr = self.prospective_result
            print(f"  -> Validation prospective : {pr['n_forecasts']} previsions")
            print(f"     RMSE prospectif = {pr['rmse_prospectif']:.1f} | "
                  f"R2 prospectif = {pr['r2_prospectif']:.4f}")
        if self.sobol_result:
            dominant = max(self.sobol_result['S1'], key=self.sobol_result['S1'].get)
            print(f"  -> Sobol : parametre dominant = {dominant} "
                  f"(S1={self.sobol_result['S1'][dominant]:.4f})")
        print(f"  -> Rapport final : {report_path}")
        print(f"  -> PIPELINE COMPLET M1->M5 TERMINE !")
        print("=" * 65)

        return validation_result


# -----------------------------------------------------------------------------
# POINT D'ENTREE : test autonome
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST AUTONOME -- Module M5 Validation")
    print("=" * 65 + "\n")

    import sys
    sys.path.insert(0, '.')
    from M1_acquisition import AcquisitionModule
    from M2_ia_prior    import IAPriorModule
    from M3_calibration import CalibrationModule

    # M1
    m1 = AcquisitionModule(
        data_path="data/usa_county_wise.csv",
        state="New York", N=19_450_000, D_I=14
    )
    df_sir, meta = m1.run(plot=False)

    # M2
    m2 = IAPriorModule(meta=meta, disease="COVID-19")
    prior_result = m2.run(df_sir=df_sir, plot=False)

    # M3
    m3 = CalibrationModule(meta=meta, prior_result=prior_result)
    calib_result = m3.run(df_sir=df_sir, plot=False)

    # M5
    m5 = ValidationModule(
        meta=meta,
        prior_result=prior_result,
        calib_result=calib_result
    )
    val_result = m5.run(df_sir=df_sir, plot=True, n_sobol=256)

    print("\n=== RESULTATS FINAUX ===")
    pr = val_result['prospective_result']
    if pr:
        print(f"  RMSE prospectif : {pr['rmse_prospectif']:.1f}")
        print(f"  R2 prospectif   : {pr['r2_prospectif']:.4f}")
    sr = val_result['sobol_result']
    if sr:
        print(f"  Sobol S1 alpha : {sr['S1']['alpha']:.4f}")
        print(f"  Sobol S1 gamma : {sr['S1']['gamma']:.4f}")
