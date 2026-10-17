"""
=============================================================================
  MODULE M3 -- CALIBRATION SEQUENTIELLE DES PARAMETRES SIR (theta(t))
=============================================================================

  Pipeline d'estimation automatique des parametres SIR (alpha, beta', gamma)
  par Intelligence Artificielle generative

  Auteur  : Henri (Master 2 - GI)
  Methode : fenetres glissantes chainees (horizon glissant) ; a l'interieur
            de chaque fenetre : Evolution Differentielle (DE) puis
            Nelder-Mead (raffinement local)

  Role de ce module :
    - Resoudre le probleme inverse SIR SUR TOUTE LA SERIE OBSERVEE, sur la
      POPULATION COMPLETE N (pas de pool effectif artificiel).
    - Plutot qu'un theta* constant unique, on determine une TRAJECTOIRE
      theta(t) = {theta_0, theta_1, ..., theta_K} : un jeu de parametres par
      fenetre temporelle, re-estime "au fur et a mesure" de l'evolution de
      l'epidemie.
    - Chaque fenetre est calibree par le meme optimiseur a deux niveaux que
      precedemment (DE puis Nelder-Mead), mais sur une TRANCHE de la serie,
      avec un budget reduit (le probleme local est plus simple : 2
      parametres sur une fenetre courte).
    - Les fenetres sont CHAINEES : l'etat (S,I,R) final de la fenetre k
      devient exactement la condition initiale de la fenetre k+1 -- la
      courbe reconstruite est donc continue par construction, sans recollage
      approximatif.

  Pourquoi ce changement (abandon du pool effectif rho, cf. versions
  precedentes de ce module) :
    Le taux d'attaque COVID reellement observe est tres faible (~1-2%), donc
    S(t)/N reste quasi constant (~0.98-0.99) sur toute la serie. Le terme
    alpha*S*I/N degenere alors en alpha*I : un SIR a parametres CONSTANTS ne
    peut produire qu'une seule cloche, jamais une dynamique multi-phases. Un
    correctif artificiel (pool effectif Neff = rho*N) permettait de forcer
    une cloche, mais ce n'est pas le bon mecanisme causal : dans la realite,
    ce ne sont PAS les susceptibles qui s'epuisent, ce sont les
    INTERVENTIONS / changements de comportement qui font chuter
    R_eff = alpha/gamma sous 1. En laissant theta VARIER dans le temps
    (fenetre par fenetre), le modele capture ce vrai mecanisme, sur la
    population N complete, sans artifice. Voir
    METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md pour la demonstration complete.

  Probleme inverse (par fenetre k) :
    theta_k* = argmin L_k(theta_k)
    ou L_k(theta_k) = sum_t w(t) [I_data(t) - I_sim(t;theta_k)]^2
                     + sum_t w(t) [C_data(t) - C_sim(t;theta_k)]^2
    avec C(t) = I(t) + R(t) (cumul confirme) et w(t) une ponderation de
    recence qui privilegie le pas REELLEMENT adopte [0,X) sur le reste du
    contexte [X,Y). Le terme sur C(t) est necessaire : I(t) seul ne suffit
    pas a identifier gamma (cf. commentaire dans __init__).

  Le systeme SIR est integre numeriquement par scipy.integrate.odeint
  (methode LSODA, adaptative) a chaque evaluation de L_k(theta_k).

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import numpy as np                          # Calculs numeriques vectorises
import matplotlib.pyplot as plt             # Visualisation des resultats
import matplotlib.dates as mdates           # Formatage des axes temporels
import pandas as pd                         # Manipulation des donnees
from scipy.integrate import odeint          # Solveur EDO (LSODA, adaptatif)
from scipy.optimize import differential_evolution, minimize  # Optimiseurs
import os
import time
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# CONSTANTES GLOBALES
# -----------------------------------------------------------------------------

# --- Fenetrage temporel de la calibration sequentielle -----------------------
# Y = fenetre de CONTEXTE utilisee pour identifier theta_k (le fit a besoin
#     d'assez de points pour separer alpha et gamma de facon fiable).
# X = pas de glissement = duree pendant laquelle theta_k est effectivement
#     "adopte" dans la trajectoire et dans la courbe reconstruite. X < Y :
#     les fenetres de contexte se chevauchent (chaque theta_k est identifie
#     sur 21 jours) mais seuls les 7 premiers jours de chaque fenetre servent
#     a construire la trajectoire finale -- schema a horizon glissant
#     (receding horizon), analogue a la commande predictive (MPC).
WINDOW_SIZE_DAYS = 10   # Y : taille de la fenetre de contexte (jours)
WINDOW_STEP_DAYS = 5    # X : pas de glissement / duree adoptee (jours)
MIN_WINDOW_DAYS  = 5    # en dessous, on ne re-fit pas : on prolonge theta_(K-1)

# Seuil de "signal degenere" : si le pic observe SUR LA FENETRE DE CONTEXTE
# est en dessous de cette fraction du pic GLOBAL, I_data y est quasi-nul (bruit
# de fond avant le vrai demarrage de l'epidemie). Dans ce cas la fonction de
# cout est PLATE (n'importe quel theta donne un cout ~0) : un fit DE/NM y
# produirait un resultat ARBITRAIRE, parfois R0<1, ce qui ferait decroitre un
# I deja infinitesimal -- un artefact numerique, pas un signal
# epidemiologique. On conserve alors le dernier regime informe plutot que de
# fitter du bruit (cf. run_sequential).
DEGENERATE_SIGNAL_FRAC = 0.01

# Parametres de l'optimiseur Evolution Differentielle -- PAR FENETRE
# (budget reduit vs. un fit global : chaque probleme local est plus simple,
#  et il y a ~T/X fenetres a calibrer)
WIN_DE_POPSIZE = 12
WIN_DE_MAXITER = 150
WIN_DE_TOL     = 1e-6
WIN_DE_SEED    = 42

# Parametres de l'optimiseur Nelder-Mead (niveau 2, raffinement local)
WIN_NM_MAXITER = 2000
NM_XATOL       = 1e-8  # Tolerance sur les parametres
NM_FATOL       = 1e-8  # Tolerance sur la fonction de cout


# -----------------------------------------------------------------------------
# CLASSE PRINCIPALE : CalibrationModule
# -----------------------------------------------------------------------------
class CalibrationModule:
    """
    Module M3 du pipeline de calibration SIR -- version sequentielle.

    Resout le probleme inverse SIR fenetre par fenetre : theta_k* = (alpha_k*,
    gamma_k*), avec continuite exacte de l'etat (S,I,R) entre fenetres
    successives. La sortie principale est une TRAJECTOIRE theta(t), pas un
    theta* unique.

    Strategie d'optimisation a deux niveaux, appliquee PAR FENETRE :
    - Niveau 1 (global)  : Evolution Differentielle -- evite les minima locaux
    - Niveau 2 (local)   : Nelder-Mead -- raffine precisement la solution

    Attributs
    ---------
    N                : int   -- taille de la population COMPLETE
    t_obs            : array -- vecteur temporel (jours)
    S_data, I_data, R_data : array -- compartiments observes (normalises /N)
    bounds_alpha, bounds_gamma, bounds_i0 : tuples -- bornes de recherche
    theta_trajectory : list[dict] -- trajectoire theta_k, k=0..K-1
    theta_star       : dict  -- DERNIER regime connu (compat M4/M5/M6)
    sol_star         : ndarray (T,3) -- reconstruction complete concatenee
    metrics          : dict  -- metriques de validation GLOBALES (RMSE, R2...)
    """

    def __init__(self, meta, prior_result):
        """
        Initialise le module M3 avec les sorties de M1 et M2.

        Parameters
        ----------
        meta         : dict -- metadonnees de M1 (N, t_obs, S_data, I_data, R_data)
        prior_result : dict -- resultats de M2 (bounds, theta_0)
        """
        # Donnees observees (issues du Module M1) -- normalisees par N
        # La normalisation evite les problemes numeriques avec de grandes
        # valeurs. N est la population COMPLETE : pas de pool effectif.
        self.N     = meta['N']
        self.t_obs = meta['t_obs'].astype(float)
        self.S_data = meta['S_data'] / meta['N']   # normalise [0,1]
        self.I_data = meta['I_data'] / meta['N']   # normalise [0,1]
        self.R_data = meta['R_data'] / meta['N']   # normalise [0,1]
        self.T     = len(self.t_obs)

        # Bornes de recherche pour (alpha, gamma) -- elargies (x0.5 / x2)
        # autour du prior M2 pour donner de la liberte a l'optimiseur, FIXES
        # pour TOUTES les fenetres (pas de recentrage sur theta_(k-1) :
        # chaque fenetre est identifiee independamment dans le meme espace de
        # recherche, ce qui evite tout biais de chainage et reste simple a
        # defendre devant un jury).
        #
        # NOTE D'IDENTIFIABILITE : ajuster (alpha, gamma) sur la seule
        # prevalence I(t) est structurellement sous-determine -- un DUO
        # (alpha eleve, gamma eleve) et un DUO plus lent peuvent reproduire
        # la MEME valeur ponctuelle de I(t) avec un "debit" I->R totalement
        # different (verifie experimentalement : sans contrainte
        # supplementaire, gamma sature ses bornes hautes et le cumul de
        # retires reconstruit explose bien au-dela du taux d'attaque reel,
        # meme quand I(t) semble bien suivi). La fonction de cout
        # (_fit_window, plus bas) inclut donc AUSSI un terme sur le cumul
        # confirme C(t)=I(t)+R(t), qui resout cette ambiguite sans avoir
        # besoin de bornes artificiellement restrictives sur gamma. Voir
        # METHODOLOGIE_CALIBRATION_SEQUENTIELLE.md, section limites.
        bounds_m2 = prior_result['bounds']
        self.bounds_alpha = (
            max(0.001, bounds_m2[0][0] * 0.5),
            min(1.500, bounds_m2[0][1] * 2.0),
        )
        self.bounds_gamma = (
            max(0.001, bounds_m2[2][0] * 0.5),
            min(0.400, bounds_m2[2][1] * 2.0),
        )

        # Bornes pour i0 (prevalence initiale, fraction de N) -- UNIQUEMENT
        # utilisees pour la toute premiere fenetre (k=0) : au-dela, l'etat
        # initial de chaque fenetre est herite par continuite de la fenetre
        # precedente, ce n'est plus un parametre libre.
        i0_guess = max(float(self.I_data[0]), 1e-9)
        lo_i0 = max(1e-9, i0_guess * 0.1)
        hi_i0 = min(0.5, max(i0_guess * 50.0, 1e-6))
        if hi_i0 <= lo_i0:
            hi_i0 = lo_i0 * 10.0 + 1e-6
        self.bounds_i0 = (lo_i0, hi_i0)

        # beta' (min_infectious_time GAMA) : NON calibre -- fixe depuis la
        # litterature (D_I du prior M2). Il est structurellement absent de
        # l'EDO SIR continue (Sobol S1=0 constate) : le calibrer n'aurait
        # aucun effet sur la dynamique. Ce point est INDEPENDANT de la
        # refonte sequentielle et reste inchange.
        self.beta_prime_fixed = int(round(prior_result['D_I_prior']))
        self.theta_0     = prior_result['theta_0']
        self.prior_R0    = prior_result['R0_prior']
        self.D_I_prior   = prior_result['D_I_prior']

        # Resultats initialises a None/vide -- remplis apres calibration
        self.theta_trajectory = []   # LA trajectoire theta(t), coeur du module
        self.theta_star  = None      # dernier regime connu (compat aval)
        self.metrics     = None
        self.sol_star    = None      # reconstruction complete concatenee
        self.n_windows   = 0

        # Compteur d'evaluations de la fonction de cout (toutes fenetres confondues)
        self.n_eval = 0

        # Metadonnees completes (state, disease, etc.) pour le graphique
        self.meta = meta

        n_windows_prevu = max(1, int(np.ceil(self.T / WINDOW_STEP_DAYS)))
        print("=" * 65)
        print("  MODULE M3 -- Calibration Sequentielle des Parametres SIR")
        print("=" * 65)
        print(f"  Population N       : {self.N:,}  (population COMPLETE)")
        print(f"  Serie               : {self.T} jours")
        print(f"  Fenetrage           : contexte={WINDOW_SIZE_DAYS}j, "
              f"pas={WINDOW_STEP_DAYS}j  (~{n_windows_prevu} fenetres prevues)")
        print(f"  Methode par fenetre : Evolution Differentielle + Nelder-Mead")
        print(f"  Point depart (M2)  : alpha={self.theta_0['alpha_0']}, "
              f"beta'={self.theta_0['beta_prime_0']}, "
              f"gamma={self.theta_0['gamma_0']}")
        print(f"  Bornes alpha        : {self.bounds_alpha}")
        print(f"  Bornes gamma        : {self.bounds_gamma}")
        print(f"  Bornes i0 (1re fen.): {self.bounds_i0}")
        print(f"  beta' fixe          : {self.beta_prime_fixed} j (litterature, non calibre)")
        print("=" * 65)

    # -------------------------------------------------------------------------
    # SOLVEUR SIR : Resolution du systeme d'equations differentielles
    # -------------------------------------------------------------------------
    def _sir_ode(self, y, t, alpha, gamma):
        """
        Definit le systeme d'equations differentielles du modele SIR.

        Le systeme SIR continu est (fractions de N, N n'apparait plus car
        tout est deja en proportion de la population COMPLETE) :
          dS/dt = -alpha * S * I
          dI/dt =  alpha * S * I  -  gamma * I
          dR/dt =  gamma * I

        avec la contrainte : S(t) + I(t) + R(t) = 1 (population fermee)

        Parameters
        ----------
        y     : list  -- [S, I, R] etat courant (fractions de N)
        t     : float -- temps courant (jours)
        alpha : float -- taux de transmission (infection_rate GAMA)
        gamma : float -- taux de guerison (recovery_rate GAMA)

        Returns
        -------
        [dS/dt, dI/dt, dR/dt] : derivees du systeme
        """
        S, I, R = y
        dS = -alpha * S * I              # S diminue par infection
        dI =  alpha * S * I - gamma * I  # I monte puis descend
        dR =  gamma * I                  # R augmente par guerison
        return [dS, dI, dR]

    def _solve_sir(self, alpha, gamma, S0, I0, R0, t_array):
        """
        Resout le systeme SIR sur la population COMPLETE N, a partir d'une
        condition initiale (S0, I0, R0) donnee (fractions de N, S0+I0+R0=1).

        Contrairement aux versions precedentes de ce module, il n'y a plus de
        pool effectif : sol*N donne directement des effectifs (personnes) sur
        la population entiere.

        Parameters
        ----------
        alpha, gamma : float -- parametres du regime courant
        S0, I0, R0   : float -- condition initiale (fractions de N, somme=1)
        t_array      : array -- vecteur temporel de la fenetre a integrer

        Returns
        -------
        sol : ndarray (len(t_array), 3) -- [S, I, R] en fraction de N
              None si la resolution echoue
        """
        try:
            sol = odeint(
                self._sir_ode,
                [S0, I0, R0],
                t_array,
                args   = (alpha, gamma),
                rtol   = 1e-7,
                atol   = 1e-9,
                mxstep = 5000
            )
            return sol
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # CALIBRATION D'UNE FENETRE : theta_k = (alpha_k, gamma_k[, i0_k])
    # -------------------------------------------------------------------------
    def _fit_window(self, k, day_start, ctx_end, S0, I0, R0, free_i0):
        """
        Calibre theta_k sur la fenetre de CONTEXTE [day_start, ctx_end) par
        Evolution Differentielle puis Nelder-Mead.

        Pour k=0 (premiere fenetre), i0 est un parametre LIBRE (calibre) :
        l'etat initial exact de l'epidemie n'est pas connu avec certitude.
        Pour k>=1, l'etat initial (S0, I0, R0) est FIXE -- herite par
        continuite exacte de la fin de la fenetre k-1 -- seuls (alpha, gamma)
        sont calibres.

        Parameters
        ----------
        k          : int   -- indice de la fenetre
        day_start  : int   -- premier jour de la fenetre de contexte
        ctx_end    : int   -- jour de fin (exclu) de la fenetre de contexte
        S0, I0, R0 : float -- condition initiale (utilisee si free_i0=False)
        free_i0    : bool  -- True uniquement pour k=0

        Returns
        -------
        dict -- {'alpha_k','gamma_k','i0_k','S0','I0','R0_0','cost_k','de_hit_bound'}
        """
        t_slice      = self.t_obs[day_start:ctx_end]
        I_data_slice  = self.I_data[day_start:ctx_end]
        # Cumul confirme C(t) = I(t) + R(t) -- terme AJOUTE a la fonction de
        # cout (voir plus bas). I(t) seul ne suffit pas a identifier gamma :
        # un duo (alpha eleve, gamma eleve) et un duo (alpha modere, gamma
        # modere) peuvent reproduire la MEME valeur ponctuelle de I(t) avec
        # un "debit" I->R totalement different (probleme classique
        # d'identifiabilite d'un systeme SIR ajuste sur la seule prevalence).
        # Le cumul C(t), lui, est monotone et directement sensible a ce
        # debit : l'ajouter au cout resout cette ambiguite sans avoir besoin
        # de bornes artificiellement restrictives sur gamma.
        C_data_slice = I_data_slice + self.R_data[day_start:ctx_end]

        # Ponderation de recence : le contexte [X, Y) au-dela du pas adopte X
        # n'est la que pour AIDER l'identifiabilite de (alpha, gamma) -- ce
        # qui compte vraiment, c'est que le modele colle bien sur le pas
        # REELLEMENT adopte [0, X). Sans cette ponderation, un compromis sur
        # tout le contexte peut degrader precisement la portion adoptee (la
        # seule qui entre dans la trajectoire chainee finale).
        adopted_len = min(WINDOW_STEP_DAYS, ctx_end - day_start)
        weights = np.ones(len(t_slice))
        weights[:adopted_len] = 3.0

        bounds = [self.bounds_alpha, self.bounds_gamma]
        if free_i0:
            bounds = bounds + [self.bounds_i0]

        def cost(theta):
            alpha, gamma = float(theta[0]), float(theta[1])
            if not (self.bounds_alpha[0] <= alpha <= self.bounds_alpha[1] and
                    self.bounds_gamma[0] <= gamma <= self.bounds_gamma[1]):
                return 1e12
            if free_i0:
                i0 = float(theta[2])
                if not (self.bounds_i0[0] <= i0 <= self.bounds_i0[1]):
                    return 1e12
                I0_w, R0_w = i0, 0.0
                S0_w = max(0.0, 1.0 - I0_w - R0_w)
            else:
                S0_w, I0_w, R0_w = S0, I0, R0

            sol = self._solve_sir(alpha, gamma, S0_w, I0_w, R0_w, t_slice)
            if sol is None:
                return 1e12

            C_sim = sol[:, 1] + sol[:, 2]
            c = float(np.sum(weights * (I_data_slice - sol[:, 1]) ** 2) +
                       np.sum(weights * (C_data_slice - C_sim) ** 2))
            self.n_eval += 1
            return c

        result_de = differential_evolution(
            func          = cost,
            bounds        = bounds,
            popsize       = WIN_DE_POPSIZE,
            maxiter       = WIN_DE_MAXITER,
            tol           = WIN_DE_TOL,
            seed          = WIN_DE_SEED,
            polish        = False,
            mutation      = (0.5, 1.5),
            recombination = 0.9,
            workers       = 1,
        )
        result_nm = minimize(
            fun     = cost,
            x0      = result_de.x,
            method  = 'Nelder-Mead',
            options = {
                'maxiter' : WIN_NM_MAXITER,
                'xatol'   : NM_XATOL,
                'fatol'   : NM_FATOL,
                'disp'    : False,
            }
        )

        alpha_k = float(result_nm.x[0])
        gamma_k = float(result_nm.x[1])

        # Diagnostic : theta_k a-t-il bute sur une borne de l'espace de
        # recherche ? (signale une fenetre potentiellement mal identifiee)
        eps = 1e-4
        de_hit_bound = (
            abs(alpha_k - self.bounds_alpha[0]) < eps or
            abs(alpha_k - self.bounds_alpha[1]) < eps or
            abs(gamma_k - self.bounds_gamma[0]) < eps or
            abs(gamma_k - self.bounds_gamma[1]) < eps
        )

        if free_i0:
            i0_k = float(result_nm.x[2])
            I0_final, R0_final = i0_k, 0.0
            S0_final = max(0.0, 1.0 - I0_final - R0_final)
        else:
            i0_k = None
            S0_final, I0_final, R0_final = S0, I0, R0

        return {
            'alpha_k'      : alpha_k,
            'gamma_k'      : gamma_k,
            'i0_k'         : i0_k,
            'S0'           : S0_final,
            'I0'           : I0_final,
            'R0_0'         : R0_final,
            'cost_k'       : float(result_nm.fun),
            'de_hit_bound' : bool(de_hit_bound),
        }

    # -------------------------------------------------------------------------
    # ORCHESTRATION DE LA CALIBRATION SEQUENTIELLE : theta(t) fenetre par fenetre
    # -------------------------------------------------------------------------
    def run_sequential(self):
        """
        Construit la trajectoire theta(t) = {theta_0, theta_1, ..., theta_K}
        en parcourant toute la serie par fenetres glissantes chainees.

        Algorithme (horizon glissant / receding-horizon) :
          1. Partir de l'etat initial derive de I_data(0).
          2. Pour chaque fenetre k : calibrer theta_k sur la fenetre de
             CONTEXTE [day, day+Y) par DE+Nelder-Mead (_fit_window).
          3. Integrer le SIR avec theta_k UNIQUEMENT sur le pas adopte
             [day, day+X) -- c'est cette portion qui entre dans la
             trajectoire et dans la courbe reconstruite finale.
          4. L'etat (S,I,R) en fin de pas devient EXACTEMENT la condition
             initiale de la fenetre k+1 : continuite garantie par
             construction (pas de recollement approche).
          5. Si le reliquat de serie est trop court pour un fit fiable
             (< MIN_WINDOW_DAYS), on NE refit PAS : on prolonge le dernier
             regime connu jusqu'a la fin de la serie plutot que de
             sur-ajuster sur trop peu de points.

        Returns
        -------
        self -- chaine les methodes
        """
        print(f"\n[M3 - Calibration sequentielle] theta(t) fenetre par fenetre...")
        t_start_total = time.time()

        day = 0
        k   = 0
        trajectory = []
        sol_chunks = []

        # Etat initial derive de la premiere observation (fraction de N)
        I0_data = max(float(self.I_data[0]), 1e-9)
        S0, I0, R0 = max(0.0, 1.0 - I0_data), I0_data, 0.0

        peak_global = float(np.max(self.I_data))
        signal_threshold = DEGENERATE_SIGNAL_FRAC * peak_global

        while day < self.T:
            ctx_end  = min(day + WINDOW_SIZE_DAYS, self.T)
            step_end = min(day + WINDOW_STEP_DAYS, self.T)
            low_signal = float(np.max(self.I_data[day:ctx_end])) < signal_threshold

            if (self.T - day) < MIN_WINDOW_DAYS and k > 0:
                # Reliquat trop court pour un refit fiable : on prolonge le
                # dernier regime connu jusqu'a la fin de la serie.
                last = trajectory[-1]
                alpha_k, gamma_k, i0_k = last['alpha_k'], last['gamma_k'], None
                cost_k, de_hit_bound, prolonged = None, False, True
                step_end = self.T
            elif low_signal:
                # Fenetre non informative (I_data quasi-nul) : on NE FIT PAS
                # (cout plat -> resultat arbitraire). On garde le dernier
                # regime informe -- le prior IA pour la toute premiere
                # fenetre, sinon le regime k-1 -- jusqu'a ce que le signal
                # redevienne exploitable.
                if k == 0:
                    alpha_k, gamma_k = self.theta_0['alpha_0'], self.theta_0['gamma_0']
                else:
                    alpha_k, gamma_k = trajectory[-1]['alpha_k'], trajectory[-1]['gamma_k']
                i0_k = None
                cost_k, de_hit_bound, prolonged = None, False, False
            else:
                free_i0 = (k == 0)
                fit = self._fit_window(k, day, ctx_end, S0, I0, R0, free_i0)
                alpha_k, gamma_k, i0_k = fit['alpha_k'], fit['gamma_k'], fit['i0_k']
                cost_k, de_hit_bound, prolonged = fit['cost_k'], fit['de_hit_bound'], False
                if free_i0:
                    S0, I0, R0 = fit['S0'], fit['I0'], fit['R0_0']

            t_step = self.t_obs[day:step_end]
            sol_step = self._solve_sir(alpha_k, gamma_k, S0, I0, R0, t_step)
            if sol_step is None:
                # Filet de securite (ne devrait pas arriver en pratique) :
                # etat maintenu constant sur le pas pour ne pas casser la
                # chaine de continuite.
                sol_step = np.tile([S0, I0, R0], (len(t_step), 1))

            sol_chunks.append(sol_step)

            S_end, I_end, R_end = [float(v) for v in sol_step[-1]]

            I_data_step = self.I_data[day:step_end]
            I_sim_step = sol_step[:, 1]
            rmse_k = float(np.sqrt(np.mean((I_data_step - I_sim_step) ** 2)) * self.N)
            ss_res = float(np.sum((I_data_step - I_sim_step) ** 2))
            ss_tot = float(np.sum((I_data_step - np.mean(I_data_step)) ** 2))
            r2_k   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            record = {
                'k'            : k,
                'day_start'    : int(day),
                'day_end_step' : int(step_end),
                'day_end_ctx'  : int(ctx_end),
                'alpha_k'      : alpha_k,
                'gamma_k'      : gamma_k,
                'R0_k'         : (alpha_k / gamma_k) if gamma_k > 0 else 0.0,
                'i0_k'         : i0_k,
                'S_start'      : float(S0),
                'I_start'      : float(I0),
                'R_start'      : float(R0),
                'S_end_step'   : S_end,
                'I_end_step'   : I_end,
                'R_end_step'   : R_end,
                'rmse_k'       : rmse_k,
                'r2_k'         : r2_k,
                'cost_k'       : cost_k,
                'de_hit_bound' : de_hit_bound,
                'prolonged'    : prolonged,
                'low_signal'   : low_signal,
            }
            trajectory.append(record)

            tag = (" [prolonge]" if prolonged else
                   " [signal faible]" if low_signal else
                   " [!borne]" if de_hit_bound else "")
            print(f"  -> Fenetre {k:>2} | jours {day:>3}-{step_end-1:<3} "
                  f"(contexte jusqu'a J{ctx_end-1:<3}) | "
                  f"alpha={alpha_k:.4f} gamma={gamma_k:.4f} "
                  f"R0={record['R0_k']:.3f} | RMSE_loc={rmse_k:,.1f} "
                  f"R2_loc={r2_k:.3f}{tag}")

            # Continuite exacte : l'etat final devient l'IC de la fenetre suivante
            S0, I0, R0 = S_end, I_end, R_end
            day = step_end
            k  += 1

        self.theta_trajectory = trajectory
        self.n_windows        = len(trajectory)
        self.sol_star         = np.concatenate(sol_chunks, axis=0)

        # Verification programmatique de la continuite aux jonctions (preuve,
        # pas seulement visuelle, que la trajectoire est bien chainee).
        max_gap = 0.0
        for i in range(1, len(trajectory)):
            prev, cur = trajectory[i - 1], trajectory[i]
            gap = max(
                abs(cur['S_start'] - prev['S_end_step']),
                abs(cur['I_start'] - prev['I_end_step']),
                abs(cur['R_start'] - prev['R_end_step']),
            )
            max_gap = max(max_gap, gap)
        assert max_gap < 1e-6, f"Discontinuite detectee aux jonctions (max_gap={max_gap:.2e})"

        # Dernier regime connu -- expose pour compatibilite M4/M5/M6 (valeur
        # par defaut / "regime courant" de fin de serie).
        last = trajectory[-1]
        self.theta_star = {
            'alpha'      : round(last['alpha_k'], 6),
            'beta_prime' : self.beta_prime_fixed,
            'gamma'      : round(last['gamma_k'], 6),
            'R0'         : round(last['R0_k'], 4),
            'cost_final' : last['cost_k'],
        }

        t_elapsed = time.time() - t_start_total
        print(f"\n  RESULTAT calibration sequentielle :")
        print(f"  -> {self.n_windows} fenetres calibrees sur {self.T} jours")
        print(f"  -> Continuite aux jonctions : OK (ecart max = {max_gap:.2e})")
        print(f"  -> Dernier regime  : alpha*={self.theta_star['alpha']:.4f} "
              f"gamma*={self.theta_star['gamma']:.4f} R0*={self.theta_star['R0']:.3f}")
        print(f"  -> Evaluations SIR : {self.n_eval:,}")
        print(f"  -> Temps total     : {t_elapsed:.1f}s")

        return self

    # -------------------------------------------------------------------------
    # ETAPE : Calcul des metriques de validation GLOBALES
    # -------------------------------------------------------------------------
    def compute_metrics(self):
        """
        Calcule les metriques de validation GLOBALES de la reconstruction
        (sur les 188 jours complets, theta(t) variable, N complet).

        RMSE (Root Mean Square Error) :
          RMSE = sqrt( (1/T) * sum((I_data - I_sim)^2) )
          Unite : cas/jour. Penalise fortement les grands ecarts.

        MAE (Mean Absolute Error) :
          MAE = (1/T) * sum(|I_data - I_sim|)
          Unite : cas/jour. Moins sensible aux valeurs extremes.

        R2 (Coefficient de determination) :
          R2 = 1 - sum((I_data - I_sim)^2) / sum((I_data - I_mean)^2)
          Sans unite. R2 = 1 : ajustement parfait. R2 > 0.85 : bon.

        Erreur sur le pic epidemique :
          Ecart en jours et en amplitude entre pic observe et pic modele.

        Returns
        -------
        self -- chaine les methodes
        """
        print(f"\n[M3 - Metriques] Calcul des metriques de validation globales...")

        if self.sol_star is None:
            print("  !! Solution non disponible -- calibration non effectuee")
            return self

        I_sim  = self.sol_star[:, 1] * self.N
        I_data  = self.I_data * self.N
        I_mean = np.mean(I_data)

        rmse = np.sqrt(np.mean((I_data - I_sim) ** 2))
        mae  = np.mean(np.abs(I_data - I_sim))

        ss_res = np.sum((I_data - I_sim) ** 2)
        ss_tot = np.sum((I_data - I_mean) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        peak_data_idx = np.argmax(I_data)
        peak_sim_idx = np.argmax(I_sim)
        peak_data_val = I_data[peak_data_idx]
        peak_sim_val = I_sim[peak_sim_idx]
        peak_day_err = abs(int(peak_data_idx) - int(peak_sim_idx))
        peak_amp_err = abs(peak_data_val - peak_sim_val)

        # Duree moyenne de la maladie (formule theorique, dernier regime connu)
        b = self.theta_star['beta_prime']
        g = self.theta_star['gamma']
        duree_moyenne = b + (1 - g) / g

        self.metrics = {
            'RMSE'          : round(float(rmse), 2),
            'MAE'           : round(float(mae), 2),
            'R2'            : round(float(r2), 4),
            'peak_day_err'  : int(peak_day_err),
            'peak_amp_err'  : round(float(peak_amp_err), 1),
            'peak_data_day'  : int(peak_data_idx),
            'peak_sim_day'  : int(peak_sim_idx),
            'peak_data_val'  : round(float(peak_data_val), 1),
            'peak_sim_val'  : round(float(peak_sim_val), 1),
            'duree_maladie' : round(float(duree_moyenne), 1),
            'n_windows'     : self.n_windows,
            'n_eval_total'  : self.n_eval,
        }

        if r2 >= 0.90:
            qualite = "EXCELLENT (R2 >= 0.90)"
        elif r2 >= 0.80:
            qualite = "BON (R2 >= 0.80)"
        elif r2 >= 0.70:
            qualite = "ACCEPTABLE (R2 >= 0.70)"
        else:
            qualite = "A AMELIORER (R2 < 0.70)"

        print(f"\n  METRIQUES DE VALIDATION (serie complete, {self.T} jours, N entier) :")
        print(f"  {'Metrique':<25} {'Valeur':<15} {'Seuil cible'}")
        print(f"  {'-'*55}")
        print(f"  {'RMSE (cas/jour)':<25} {rmse:<15.1f} {'< 2000'}")
        print(f"  {'MAE  (cas/jour)':<25} {mae:<15.1f} {'< 1500'}")
        print(f"  {'R2':<25} {r2:<15.4f} {'> 0.85'}")
        print(f"  {'Erreur pic (jours)':<25} {peak_day_err:<15} {'< 7 jours'}")
        print(f"  {'Erreur amplitude':<25} {peak_amp_err:<15.1f} {'< 10000'}")
        print(f"  {'Duree moy. maladie':<25} {duree_moyenne:<15.1f} {'jours'}")
        print(f"  {'-'*55}")
        print(f"  Qualite globale : {qualite}")

        return self

    # -------------------------------------------------------------------------
    # VISUALISATION : Courbes de calibration
    # -------------------------------------------------------------------------
    def plot_calibration(self, df_sir,
                          save_path="outputs/M3_calibration.png"):
        """
        Genere les courbes de calibration pour le memoire.

        Quatre graphiques sont produits :

        1. I(t) observe vs modele reconstruit -- courbe principale
        2. Residus (I_data - I_sim) avec bande +/-RMSE
        3. APERCU de la trajectoire theta(t) : alpha_k(t) et gamma_k(t) en
           escalier au fil des fenetres (teaser -- la version complete et
           annotee vit dans M6_comparaison.py, FIG2)
        4. Tableau des resultats -- resume numerique complet

        Parameters
        ----------
        df_sir    : pd.DataFrame -- donnees SIR du Module M1
        save_path : str          -- chemin de sauvegarde
        """
        os.makedirs("outputs", exist_ok=True)

        if self.sol_star is None:
            print("  !! Pas de solution -- graphique non genere")
            return

        has_date = 'Date' in df_sir.columns
        if has_date:
            dates = df_sir['Date']
        else:
            dates = pd.Series(np.asarray(df_sir['t'].values, dtype=float))

        def _fmt_time_axis(ax):
            if has_date:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
            else:
                ax.set_xlabel('Jour (depuis le debut de la serie)')

        def _x_at(day_idx):
            """Position sur l'axe temporel (date ou jour) pour un indice de jour."""
            idx = min(max(int(day_idx), 0), len(dates) - 1)
            return dates.iloc[idx]

        ts = self.theta_star
        mt = self.metrics

        S_sim = self.sol_star[:, 0] * self.N
        I_sim = self.sol_star[:, 1] * self.N
        R_sim = self.sol_star[:, 2] * self.N

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f"Module M3 -- Calibration Sequentielle SIR | "
            f"{self.meta.get('state', 'Unknown')} | {self.meta.get('disease', 'COVID-19')}\n"
            f"{self.n_windows} fenetres | dernier regime : "
            f"alpha*={ts['alpha']:.4f}  beta'*={ts['beta_prime']}j  "
            f"gamma*={ts['gamma']:.4f}  R0*={ts['R0']:.3f}",
            fontsize=14, fontweight='bold'
        )

        # -- Graphique 1 : I(t) donnees vs simule (principal) ----
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(dates, self.I_data * self.N,
                 color='red', linewidth=2.5, label='I(t) donnees')
        ax1.plot(dates, I_sim,
                 color='blue', linewidth=2, linestyle='--',
                 label=f'I(t) simule theta(t) (R2={mt["R2"]:.3f})')
        ax1.fill_between(dates, self.I_data * self.N, I_sim,
                         alpha=0.15, color='purple', label='Ecart residuel')
        ax1.axvline(_x_at(mt['peak_data_day']),
                    color='red', linestyle=':', alpha=0.7,
                    label=f'Pic donnees (J{mt["peak_data_day"]})')
        ax1.axvline(_x_at(mt['peak_sim_day']),
                    color='blue', linestyle=':', alpha=0.7,
                    label=f'Pic simule (J{mt["peak_sim_day"]})')
        ax1.set_title('Infectes actifs I(t) -- Donnees vs Simule (theta(t), N entier)',
                      fontsize=11)
        ax1.set_ylabel('Infectes actifs')
        ax1.legend(fontsize=8, loc='upper right')
        _fmt_time_axis(ax1)
        ax1.grid(True, alpha=0.3)

        # -- Graphique 2 : Residus (I_data - I_sim) avec bande +/-RMSE --
        ax2 = fig.add_subplot(2, 2, 2)
        residus = self.I_data * self.N - I_sim
        rmse    = mt['RMSE']
        ax2.axhline(0, color='black', linewidth=1)
        ax2.axhspan(-rmse, rmse, color='green', alpha=0.12,
                    label=f'Bande +/- RMSE ({rmse:,.0f})')
        ax2.plot(dates, residus, color='purple', linewidth=1.8,
                 label='Residu I(t) = donnees - simule')
        ax2.fill_between(dates, residus, 0, color='purple', alpha=0.15)
        ax2.set_title('Residus de calibration -- ecart donnees vs simule',
                      fontsize=11)
        ax2.set_ylabel('Residu (infectes actifs)')
        ax2.legend(fontsize=8, loc='upper right')
        _fmt_time_axis(ax2)
        ax2.grid(True, alpha=0.3)

        # -- Graphique 3 : Apercu de la trajectoire theta(t) -----
        ax3 = fig.add_subplot(2, 2, 3)
        if self.theta_trajectory:
            xs = [_x_at(tr['day_start']) for tr in self.theta_trajectory]
            xs.append(_x_at(self.theta_trajectory[-1]['day_end_step']))
            alphas = [tr['alpha_k'] for tr in self.theta_trajectory]
            gammas = [tr['gamma_k'] for tr in self.theta_trajectory]
            ax3.step(xs, alphas + [alphas[-1]], where='post',
                     color='darkorange', linewidth=2, label='alpha_k(t)')
            ax3.step(xs, gammas + [gammas[-1]], where='post',
                     color='teal', linewidth=2, label='gamma_k(t)')
            ax3.set_title(f"Trajectoire theta(t) -- {self.n_windows} fenetres "
                          f"(voir FIG2 de M6_comparaison.py pour la version complete)",
                          fontsize=10.5)
            ax3.set_ylabel('Valeur du parametre')
            ax3.legend(fontsize=9, loc='best')
            _fmt_time_axis(ax3)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Trajectoire non disponible',
                     ha='center', va='center', transform=ax3.transAxes)

        # -- Graphique 4 : Tableau des resultats -----------------
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        peak_ref = max(mt['peak_data_val'], 1.0)

        table_data = [
            ['Parametre', 'Prior M2', 'Dernier regime', 'Variation'],
            ['alpha (infection_rate)',
             f"{self.theta_0['alpha_0']:.4f}",
             f"{ts['alpha']:.4f}",
             f"{(ts['alpha']-self.theta_0['alpha_0'])/self.theta_0['alpha_0']*100:+.1f}%"],
            ["beta' (min_inf_time)",
             f"{self.theta_0['beta_prime_0']} j",
             f"{ts['beta_prime']} j",
             'fixe'],
            ['gamma (recovery_rate)',
             f"{self.theta_0['gamma_0']:.4f}",
             f"{ts['gamma']:.4f}",
             f"{(ts['gamma']-self.theta_0['gamma_0'])/self.theta_0['gamma_0']*100:+.1f}%"],
            ['R0 = alpha/gamma',
             f"{self.prior_R0:.3f}",
             f"{ts['R0']:.3f}",
             f"{(ts['R0']-self.prior_R0)/self.prior_R0*100:+.1f}%"],
            ['Nb. fenetres theta_k', '--', f"{self.n_windows}", '--'],
            ['', '', '', ''],
            ['Metrique', 'Valeur', 'Seuil', 'Statut'],
            ['RMSE (% pic)', f"{100*mt['RMSE']/peak_ref:.1f}%", '< 5%',
             'OK' if mt['RMSE'] < 0.05 * peak_ref else 'NOK'],
            ['MAE  (% pic)', f"{100*mt['MAE']/peak_ref:.1f}%", '< 4%',
             'OK' if mt['MAE'] < 0.04 * peak_ref else 'NOK'],
            ['R2', f"{mt['R2']:.4f}", '> 0.95',
             'OK' if mt['R2'] > 0.95 else 'NOK'],
            ['Erreur pic (j)', f"{mt['peak_day_err']}", '< 7',
             'OK' if mt['peak_day_err'] < 7 else 'NOK'],
            ['Duree maladie', f"{mt['duree_maladie']:.1f} j", '--', '--'],
        ]

        table = ax4.table(
            cellText  = table_data[1:],
            colLabels = table_data[0],
            loc       = 'center',
            cellLoc   = 'center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        for j in range(4):
            table[0, j].set_facecolor('#1F4E79')
            table[0, j].set_text_props(color='white', fontweight='bold')

        table[7, 0].set_facecolor('#D0D0D0')

        ax4.set_title('Resume de la calibration', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  -> Graphique sauvegarde : {save_path}")

    # -------------------------------------------------------------------------
    # METHODE PRINCIPALE : run()
    # -------------------------------------------------------------------------
    def run(self, df_sir, plot=True, save_path="outputs/M3_calibration.png"):
        """
        Orchestre toutes les etapes du Module M3 en sequence.

        Etapes :
        1. Calibration sequentielle theta(t) par fenetres glissantes chainees
        2. Calcul des metriques de validation globales (RMSE, MAE, R2)
        3. Visualisation des courbes de calibration (optionnel)

        Parameters
        ----------
        df_sir : pd.DataFrame -- donnees SIR issues du Module M1
        plot   : bool         -- si True, genere les graphiques

        Returns
        -------
        calib_result : dict -- theta(t) + metriques pour M4/M5/M6
        """
        print(f"\n  Demarrage de la calibration sequentielle...")
        t_total = time.time()

        self.run_sequential()
        self.compute_metrics()

        if plot:
            self.plot_calibration(df_sir, save_path=save_path)

        # Dictionnaire de sortie pour M4, M5, M6
        calib_result = {
            # Dernier regime connu -- valeur par defaut / "regime courant"
            'alpha_star'      : self.theta_star['alpha'],
            'beta_prime_star' : self.theta_star['beta_prime'],
            'gamma_star'      : self.theta_star['gamma'],
            'R0_star'         : self.theta_star['R0'],

            # Comparaison prior vs dernier regime
            'alpha_prior'     : self.theta_0['alpha_0'],
            'beta_prime_prior': self.theta_0['beta_prime_0'],
            'gamma_prior'     : self.theta_0['gamma_0'],
            'R0_prior'        : self.prior_R0,

            # Metriques de validation GLOBALES (serie complete, N entier)
            'RMSE'            : self.metrics['RMSE'],
            'MAE'             : self.metrics['MAE'],
            'R2'              : self.metrics['R2'],
            'peak_day_err'    : self.metrics['peak_day_err'],
            'peak_amp_err'    : self.metrics['peak_amp_err'],
            'duree_maladie'   : self.metrics['duree_maladie'],

            # Solution SIR complete (concatenation des fenetres, continue)
            'sol_star'        : self.sol_star,
            'n_eval'          : self.n_eval,

            # LA trajectoire theta(t) -- coeur de la nouvelle methodologie
            'theta_trajectory'  : self.theta_trajectory,
            'n_windows'         : self.n_windows,
            'window_size_days'  : WINDOW_SIZE_DAYS,
            'window_step_days'  : WINDOW_STEP_DAYS,

            # Metadonnees
            'disease'         : self.meta.get('disease', 'COVID-19'),
            'state'           : self.meta.get('state', 'Unknown'),
            'N'               : self.N,
        }

        t_elapsed = time.time() - t_total

        print("\n" + "=" * 65)
        print("  MODULE M3 -- TERMINE AVEC SUCCES")
        print("=" * 65)
        print(f"  {calib_result['n_windows']} regimes theta_k calibres sur {self.T} jours")
        print(f"  DERNIER REGIME (a injecter dans GAMA par defaut) :")
        print(f"  -> infection_rate      = {calib_result['alpha_star']:.6f}")
        print(f"  -> min_infectious_time = {calib_result['beta_prime_star']} ticks")
        print(f"  -> recovery_rate       = {calib_result['gamma_star']:.6f}")
        print(f"  -> R0 estime           = {calib_result['R0_star']:.4f}")
        print(f"  VALIDATION GLOBALE (serie complete, N entier) :")
        print(f"  -> RMSE = {calib_result['RMSE']:.1f} cas/jour")
        print(f"  -> R2   = {calib_result['R2']:.4f}")
        print(f"  -> Temps total : {t_elapsed:.1f}s")
        print(f"  -> Pret pour Module M4 (calendrier GAMA piecewise)")
        print("=" * 65)

        return calib_result


# -----------------------------------------------------------------------------
# POINT D'ENTREE : test autonome
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST AUTONOME -- Module M3 Calibration Sequentielle")
    print("=" * 65 + "\n")

    import sys
    sys.path.insert(0, '.')
    from M1_acquisition import AcquisitionModule
    from M2_ia_prior    import IAPriorModule

    # Execution M1
    m1 = AcquisitionModule(
        data_path = "data/usa_county_wise.csv",
        state     = "New York",
        N         = 19_450_000,
        D_I       = 14
    )
    df_sir, meta = m1.run(plot=False)

    # Execution M2
    m2           = IAPriorModule(meta=meta, disease="COVID-19")
    prior_result = m2.run(df_sir=df_sir, plot=False)

    # Execution M3
    m3           = CalibrationModule(meta=meta, prior_result=prior_result)
    calib_result = m3.run(df_sir=df_sir, plot=True)

    # Affichage final
    print("\n=== DERNIER REGIME theta_K POUR GAMA (valeur par defaut) ===")
    print(f"  infection_rate      <- {calib_result['alpha_star']}")
    print(f"  min_infectious_time <- {calib_result['beta_prime_star']}")
    print(f"  recovery_rate       <- {calib_result['gamma_star']}")
    print(f"  R0 estime           =  {calib_result['R0_star']}")
    print(f"\n  RMSE = {calib_result['RMSE']} cas/jour")
    print(f"  R2   = {calib_result['R2']}")
    print(f"  Nb. fenetres theta_k = {calib_result['n_windows']}")
    print(f"\n-> Module M3 valide -- pret pour M4 (calendrier piecewise GAML)")
