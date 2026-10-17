"""
=============================================================================
  MODULE M4 -- GENERATION AUTOMATIQUE DU CALENDRIER GAMA (PIECEWISE)
=============================================================================

  Pipeline d'estimation automatique des parametres SIR (alpha, beta', gamma)
  par Intelligence Artificielle generative

  Auteur  : Henri (Master 2 Genie Logiciel)

  Role de ce module :
    - Prendre la TRAJECTOIRE theta(t) calibree par M3 (une sequence de
      regimes theta_k = (alpha_k, gamma_k), un par fenetre temporelle)
    - Generer un CALENDRIER de regimes dans le fichier GAML : GAMA doit
      rejouer la meme dynamique multi-phases que la reconstruction Python,
      pas un theta constant unique.
    - Generer le fichier de configuration DSL (.sir) etendu avec la meme
      liste de regimes.
    - Ces fichiers sont directement injectables dans GAMA sans aucune
      modification manuelle.

  Pourquoi un calendrier plutot qu'un theta unique ?
    Depuis la refonte sequentielle (abandon du pool effectif rho, calibration
    par fenetres glissantes chainees sur la population COMPLETE), le pipeline
    ne produit plus un theta* constant mais une trajectoire theta(t). Pour que
    la simulation GAMA reproduise fidelement la meme dynamique que la
    reconstruction Python (M6), elle doit basculer (infection_rate,
    recovery_rate) aux memes dates de transition. C'est le mecanisme
    implemente ici : trois listes paralleles (jour de bascule, alpha, gamma)
    + un reflex qui met a jour les parametres courants selon le cycle GAMA.

  Apport scientifique :
    Ce module ferme la boucle du pipeline IDM (Ingenierie Dirigee par les
    Modeles) : la trajectoire theta(t) estimee automatiquement par
    calibration numerique sequentielle est transformee en un calendrier
    GAML directement exploitable par le simulateur GAMA -- demonstration
    concrete de l'approche MDE appliquee a l'epidemiologie computationnelle,
    et cette fois fidele a la dynamique temporelle reelle de l'epidemie.

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os                   # Gestion des fichiers et dossiers
import json                 # Serialisation des resultats
from datetime import datetime  # Horodatage de la generation
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# CLASSE PRINCIPALE : GAMLGenerationModule
# -----------------------------------------------------------------------------
class GAMLGenerationModule:
    """
    Module M4 du pipeline de calibration SIR.

    Genere automatiquement les fichiers GAML et DSL contenant le CALENDRIER
    de regimes theta(t) (un par fenetre temporelle), prets a etre injectes
    dans GAMA.

    Attributs
    ---------
    calib_result : dict -- resultats de M3 (theta_trajectory, dernier regime, metriques)
    output_dir   : str  -- dossier de sortie des fichiers generes
    timestamp    : str  -- horodatage de la generation
    """

    def __init__(self, calib_result, output_dir="outputs"):
        """
        Initialise le module M4 avec les resultats de M3.

        Parameters
        ----------
        calib_result : dict -- sortie de M3.run() (theta_trajectory incluse)
        output_dir   : str  -- dossier de destination des fichiers
        """
        self.calib_result = calib_result
        self.output_dir   = output_dir
        self.timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.theta_trajectory = calib_result.get('theta_trajectory', [])

        os.makedirs(output_dir, exist_ok=True)

        print("=" * 65)
        print("  MODULE M4 -- Generation du Calendrier GAMA (piecewise)")
        print("=" * 65)
        print(f"  Regimes theta_k a injecter : {len(self.theta_trajectory)}")
        print(f"  -> beta' (min_infectious_time) = "
              f"{calib_result['beta_prime_star']} ticks  (fixe, litterature)")
        print(f"  -> dernier regime : alpha={calib_result['alpha_star']:.6f}  "
              f"gamma={calib_result['gamma_star']:.6f}  "
              f"R0={calib_result['R0_star']:.4f}")
        print(f"  -> RMSE validation (globale) = "
              f"{calib_result['RMSE']:.1f} cas/jour")
        print(f"  -> R2   validation (globale) = "
              f"{calib_result['R2']:.4f}")
        print("=" * 65)

    # -------------------------------------------------------------------------
    # UTILITAIRE : construction des listes paralleles du calendrier
    # -------------------------------------------------------------------------
    def _regime_lists(self):
        """
        Construit les trois listes paralleles (jour de bascule, alpha_k,
        gamma_k) a partir de la trajectoire theta(t) de M3, ainsi qu'une
        liste annotee (R0_k, jour de fin) pour les commentaires/JSON.

        Returns
        -------
        starts, alphas, gammas, r0s, ends : listes paralleles (meme longueur)
        """
        traj = self.theta_trajectory
        starts = [int(tr['day_start'])      for tr in traj]
        ends   = [int(tr['day_end_step'])   for tr in traj]
        alphas = [round(float(tr['alpha_k']), 6) for tr in traj]
        gammas = [round(float(tr['gamma_k']), 6) for tr in traj]
        r0s    = [round(float(tr['R0_k']),   4) for tr in traj]
        return starts, alphas, gammas, r0s, ends

    # -------------------------------------------------------------------------
    # ETAPE 1 : Generation du fichier GAML de parametres (calendrier piecewise)
    # -------------------------------------------------------------------------
    def generate_gaml_parameters(self):
        """
        Genere le fichier GAML contenant le CALENDRIER de regimes calibres.

        Ce fichier est concu pour etre importe directement dans
        la simulation SIRMobiliteflux.gaml par la directive :
            import "params_sir_calibrated.gaml"

        Il surcharge les valeurs manuelles actuelles :
            float infection_rate  <- 0.1     // MANUEL -- REMPLACE
            int   min_infectious_time <- 10  // MANUEL -- REMPLACE
            float recovery_rate   <- 0.05    // MANUEL -- REMPLACE

        Par un CALENDRIER de regimes (infection_rate, recovery_rate) qui
        basculent au fil de la simulation, aux memes jours que la trajectoire
        theta(t) reconstruite par M3/M6 -- pas un triplet constant unique.

        Returns
        -------
        filepath : str -- chemin du fichier GAML genere
        """
        print(f"\n[M4 - Etape 1] Generation du fichier GAML (calendrier piecewise)...")

        cr = self.calib_result
        starts, alphas, gammas, r0s, ends = self._regime_lists()
        n = len(starts)

        if cr['R2'] >= 0.90:
            qualite = "EXCELLENT"
        elif cr['R2'] >= 0.80:
            qualite = "BON"
        elif cr['R2'] >= 0.70:
            qualite = "ACCEPTABLE"
        else:
            qualite = "LIMITE -- voir notes"

        gaml_starts  = "[" + ", ".join(str(s) for s in starts) + "]"
        gaml_alphas  = "[" + ", ".join(f"{a:.6f}" for a in alphas) + "]"
        gaml_gammas  = "[" + ", ".join(f"{g:.6f}" for g in gammas) + "]"

        regimes_comment_lines = "\n".join(
            f" *   regime {k:>2} : jours {starts[k]:>3}-{ends[k]-1:<3} | "
            f"alpha={alphas[k]:.4f}  gamma={gammas[k]:.4f}  R0={r0s[k]:.3f}"
            for k in range(n)
        )

        gaml_content = f"""/**
 * ==========================================================================
 * CALENDRIER DE PARAMETRES SIR theta(t) -- GENERE AUTOMATIQUEMENT
 * ==========================================================================
 *
 * Pipeline : Estimation automatique sequentielle des parametres SIR par IA
 *            generative (calibration par fenetres glissantes chainees,
 *            population COMPLETE -- pas de pool effectif)
 * Auteur   : Henri (Master 2 Genie Logiciel)
 * Date     : {self.timestamp}
 *
 * Maladie  : {cr.get('disease', 'COVID-19')}
 * Region   : {cr.get('state',   'New York')}
 * Pop. N   : {cr.get('N', 19450000):,} habitants (population COMPLETE)
 *
 * Methode de calibration :
 *   {n} fenetres theta_k, chacune calibree par Evolution Differentielle
 *   (exploration globale) puis Nelder-Mead (raffinement local), avec
 *   continuite exacte de l'etat (S,I,R) entre fenetres successives.
 *
 * Calendrier de regimes (jour de bascule -> alpha_k, gamma_k, R0_k) :
{regimes_comment_lines}
 *
 * Metriques de validation (reconstruction globale, {cr.get('N', 0):,} hab.) :
 *   RMSE = {cr['RMSE']:.1f} cas/jour
 *   MAE  = {cr['MAE']:.1f} cas/jour
 *   R2   = {cr['R2']:.4f}
 *   Qualite : {qualite}
 *   Erreur pic = {cr['peak_day_err']} jours
 *
 * UTILISATION :
 *   Ajoutez en tete de SIRMobiliteflux.gaml :
 *       import "params_sir_calibrated.gaml"
 *   Puis supprimez ou commentez les lignes manuelles :
 *       // float infection_rate <- 0.1;
 *       // int   min_infectious_time <- 10;
 *       // float recovery_rate <- 0.05;
 *   Le reflex maj_regime_theta bascule automatiquement infection_rate et
 *   recovery_rate au fil de la simulation, aux jours indiques ci-dessus.
 * ==========================================================================
 */

model SIRParametresCalibre

global {{

    // -----------------------------------------------------------------------
    // CALENDRIER DE REGIMES theta(t) -- listes paralleles
    // regime_start_day[i] = jour (cycle GAMA) a partir duquel le regime i
    // s'applique. regime_infection_rate[i] / regime_recovery_rate[i] = les
    // valeurs (alpha_i, gamma_i) calibrees sur la fenetre i.
    // -----------------------------------------------------------------------
    list<int>   regime_start_day      <- {gaml_starts};
    list<float> regime_infection_rate <- {gaml_alphas};
    list<float> regime_recovery_rate  <- {gaml_gammas};
    int         current_regime_index  <- 0;

    /**
     * infection_rate / recovery_rate : ETAT MUTABLE, reassigne par le reflex
     * maj_regime_theta ci-dessous a chaque bascule de regime. Valeur
     * initiale = regime 0 (premiere fenetre calibree).
     */
    float infection_rate <- regime_infection_rate[0];
    float recovery_rate  <- regime_recovery_rate[0];

    /**
     * beta'* = min_infectious_time (duree minimale infectieuse)
     * NON calibre -- fixe depuis la litterature (structurellement absent de
     * l'EDO SIR continue, cf. analyse de Sobol M5). Constant sur toute la
     * simulation, contrairement a infection_rate/recovery_rate.
     * Prior IA : {cr['beta_prime_prior']} ticks | Fixe : {cr['beta_prime_star']} ticks
     */
    int min_infectious_time <- {cr['beta_prime_star']};

    // -----------------------------------------------------------------------
    // BASCULE DE REGIME -- execute a chaque cycle, change infection_rate et
    // recovery_rate des que le cycle courant atteint le jour de bascule
    // suivant. Reproduit fidelement la trajectoire theta(t) calibree.
    // -----------------------------------------------------------------------
    reflex maj_regime_theta when: cycle > 0 {{
        int idx <- current_regime_index;
        loop i from: 0 to: length(regime_start_day) - 1 {{
            if (cycle >= regime_start_day[i]) {{
                idx <- i;
            }}
        }}
        if (idx != current_regime_index) {{
            current_regime_index <- idx;
            infection_rate <- regime_infection_rate[idx];
            recovery_rate  <- regime_recovery_rate[idx];
            write "Bascule regime theta_" + idx + " (jour " + cycle + ") -> "
                + "alpha=" + infection_rate + " gamma=" + recovery_rate;
        }}
    }}

    // -----------------------------------------------------------------------
    // PARAMETRES DERIVES (dernier regime -- reference / affichage)
    // -----------------------------------------------------------------------

    /**
     * R0* = Nombre de reproduction de base du DERNIER regime calibre
     * R0 = alpha_K / gamma_K = {cr['alpha_star']:.6f} / {cr['gamma_star']:.6f}
     * R0 > 1 : epidemie se propage | R0 < 1 : epidemie s'eteint
     * NOTE : R0(t) varie regime par regime -- voir regime_infection_rate /
     * regime_recovery_rate ci-dessus pour la trajectoire complete.
     */
    float R0_calibrated <- {cr['R0_star']:.6f};

    /**
     * Duree moyenne de la maladie (theorique, dernier regime)
     * E[duree] = beta'* + (1 - gamma_K) / gamma_K
     *          = {cr['beta_prime_star']} + (1 - {cr['gamma_star']:.4f}) / {cr['gamma_star']:.4f}
     *          = {cr['duree_maladie']:.1f} jours
     */
    float mean_disease_duration <- {cr['duree_maladie']:.4f};

    // -----------------------------------------------------------------------
    // METADONNEES DE CALIBRATION (pour tracabilite et reproductibilite)
    // -----------------------------------------------------------------------
    string calibration_method  <- "Calibration sequentielle theta(t) : "
                                 + "fenetres glissantes chainees, DE + Nelder-Mead";
    string calibration_disease <- "{cr.get('disease', 'COVID-19')}";
    string calibration_region  <- "{cr.get('state',   'New York')}";
    int    calibration_n_regimes <- {n};
    float  calibration_RMSE    <- {cr['RMSE']:.2f};
    float  calibration_R2      <- {cr['R2']:.6f};
    string calibration_date    <- "{self.timestamp}";

}}
"""
        filepath = os.path.join(self.output_dir, "params_sir_calibrated.gaml")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(gaml_content)

        print(f"  -> Fichier GAML genere : {filepath}")
        print(f"  -> {n} regimes theta_k dans le calendrier")
        print(f"  -> Taille : {len(gaml_content)} caracteres")
        return filepath

    # -------------------------------------------------------------------------
    # ETAPE 2 : Generation du fichier de configuration DSL (.sir)
    # -------------------------------------------------------------------------
    def generate_dsl_config(self):
        """
        Genere le fichier de configuration DSL au format .sir, etendu avec
        une section repetee `regimes { ... }` -- un bloc par fenetre theta_k
        de la trajectoire calibree.

        NOTE : si la grammaire SIRConfig.xtext est validee dans un projet
        Eclipse externe a ce depot, elle devra etre etendue separement pour
        accepter la section `regimes` -- ce fichier .sir sert de reference
        de la structure attendue.

        Returns
        -------
        filepath : str -- chemin du fichier DSL genere
        """
        print(f"\n[M4 - Etape 2] Generation du fichier DSL (.sir, calendrier)...")

        cr = self.calib_result
        starts, alphas, gammas, r0s, ends = self._regime_lists()

        regimes_block = "\n".join(
            f"    regime {{ index: {k}  day_start: {starts[k]}  day_end: {ends[k]}  "
            f"alpha: {alphas[k]:.6f}  gamma: {gammas[k]:.6f}  r0: {r0s[k]:.4f} }}"
            for k in range(len(starts))
        )

        dsl_content = f"""/*
 * Fichier de configuration SIR -- Format DSL (SIRConfig.xtext)
 * Genere automatiquement par le Module M4 du pipeline de calibration
 * Date : {self.timestamp}
 *
 * Calendrier theta(t) : {len(starts)} regimes successifs (voir bloc
 * `regimes` ci-dessous). beta (min_infectious_time) reste constant.
 */

sir-config COVID19_NewYork_2020 {{

    // Identification de la maladie et de la source de donnees
    disease : "{cr.get('disease', 'COVID-19')}"
    source  : "JHU CSSE COVID-19 Dataset -- USA County-Wise"
    region  : "{cr.get('state', 'New York')}"
    period  : "2020-01-22 to 2020-07-27"

    // beta = min_infectious_time = duree minimale infectieuse (jours),
    // FIXE (non calibre) sur toute la simulation
    beta   : {cr['beta_prime_star']}

    // Methode de calibration utilisee
    method : SEQUENTIAL_WINDOWS_DE_NM

    // Metriques de validation (reconstruction globale)
    rmse   : {cr['RMSE']:.2f}
    r2     : {cr['R2']:.6f}

    // Calendrier de regimes theta_k = (alpha_k, gamma_k) -- un bloc par
    // fenetre temporelle calibree, ordonne chronologiquement
    regimes {{
{regimes_block}
    }}
}}
"""
        filepath = os.path.join(self.output_dir, "COVID19_NewYork.sir")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dsl_content)

        print(f"  -> Fichier DSL genere  : {filepath}")
        return filepath

    # -------------------------------------------------------------------------
    # ETAPE 3 : Generation du rapport JSON complet
    # -------------------------------------------------------------------------
    def generate_json_report(self):
        """
        Genere un rapport JSON complet de la calibration, incluant le
        calendrier de regimes complet (regime_calendar) pour tracabilite.

        Returns
        -------
        filepath : str -- chemin du fichier JSON genere
        """
        print(f"\n[M4 - Etape 3] Generation du rapport JSON...")

        cr = self.calib_result
        starts, alphas, gammas, r0s, ends = self._regime_lists()

        regime_calendar = [
            {
                "index"      : k,
                "day_start"  : starts[k],
                "day_end"    : ends[k],
                "alpha"      : alphas[k],
                "gamma"      : gammas[k],
                "R0"         : r0s[k],
            }
            for k in range(len(starts))
        ]

        report = {
            "metadata" : {
                "timestamp"   : self.timestamp,
                "pipeline"    : "SIR Calibration Pipeline v2.0 -- theta(t) sequentiel",
                "author"      : "Henri -- Master 2 Genie Logiciel",
                "disease"     : cr.get('disease', 'COVID-19'),
                "region"      : cr.get('state',   'New York'),
                "population"  : cr.get('N',        19450000),
                "data_source" : "JHU CSSE COVID-19 Dataset",
                "method"      : "Fenetres glissantes chainees + DE/Nelder-Mead par fenetre"
            },
            "regime_calendar" : regime_calendar,
            "n_regimes"       : len(regime_calendar),
            "last_regime" : {
                "alpha_star"      : cr['alpha_star'],
                "beta_prime_star" : cr['beta_prime_star'],
                "gamma_star"      : cr['gamma_star'],
                "R0_star"         : cr['R0_star'],
                "mean_duration"   : cr['duree_maladie']
            },
            "first_regime_prior_comparison" : {
                "alpha_prior"      : cr['alpha_prior'],
                "beta_prime_prior" : cr['beta_prime_prior'],
                "gamma_prior"      : cr['gamma_prior'],
                "R0_prior"         : cr['R0_prior']
            },
            "validation_metrics" : {
                "RMSE"         : cr['RMSE'],
                "MAE"          : cr['MAE'],
                "R2"           : cr['R2'],
                "peak_day_err" : cr['peak_day_err'],
                "peak_amp_err" : cr['peak_amp_err']
            },
            "gama_injection" : {
                "note"                : "calendrier piecewise complet -- voir regime_calendar",
                "infection_rate"      : cr['alpha_star'],
                "min_infectious_time" : cr['beta_prime_star'],
                "recovery_rate"       : cr['gamma_star'],
                "import_directive"    : 'import "params_sir_calibrated.gaml"'
            }
        }

        filepath = os.path.join(self.output_dir, "calibration_report.json")
        import numpy as np
        def convert_numpy(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(i) for i in d]
            else:
                return convert_numpy(d)
        report_clean = clean_dict(report)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_clean, f, indent=4, ensure_ascii=False)

        print(f"  -> Rapport JSON genere : {filepath}")
        return filepath

    # -------------------------------------------------------------------------
    # ETAPE 4 : Affichage des instructions d'integration dans GAMA
    # -------------------------------------------------------------------------
    def show_integration_instructions(self):
        """
        Affiche les instructions precises pour integrer le calendrier de
        regimes calibres dans la simulation GAMA existante.
        """
        print(f"\n[M4 - Etape 4] Instructions d'integration dans GAMA...")

        cr = self.calib_result
        n = len(self.theta_trajectory)

        print(f"""
  ================================================================
  INSTRUCTIONS D'INTEGRATION DANS SIRMobiliteflux.gaml
  ================================================================

  Le fichier genere ({n} regimes theta_k) EXPOSE infection_rate et
  recovery_rate comme des variables MUTABLES, mises a jour automatiquement
  par le reflex `maj_regime_theta` a chaque bascule de regime.

  -------------------------------------------------------
  Import du fichier genere (recommande) :
  -------------------------------------------------------
  Ajoutez en TETE de SIRMobiliteflux.gaml (ligne 1) :

      import "params_sir_calibrated.gaml"

  Puis COMMENTEZ les lignes manuelles dans le bloc global :
      // float infection_rate     <- 0.1;      // remplace par le calendrier
      // int   min_infectious_time <- 10;      // remplace par calibration (fixe)
      // float recovery_rate      <- 0.05;     // remplace par le calendrier

  -------------------------------------------------------
  VERIFICATION APRES INTEGRATION :
  -------------------------------------------------------
  Dans la console GAMA, vous devriez voir au demarrage :
    infection_rate      = {cr['alpha_star']:.6f}   (regime 0, evolue ensuite)
    min_infectious_time = {cr['beta_prime_star']}                (fixe)
    recovery_rate       = {cr['gamma_star']:.6f}   (regime 0, evolue ensuite)
    R0_calibrated        = {cr['R0_star']:.4f}

  Puis, a chaque jour de bascule (regime_start_day), un message
  "Bascule regime theta_k (jour ...) -> alpha=... gamma=..." doit
  s'afficher dans la console -- c'est la preuve que GAMA rejoue bien la
  meme trajectoire theta(t) que la reconstruction Python (M6, FIG2).
  ================================================================
        """)

    # -------------------------------------------------------------------------
    # METHODE PRINCIPALE : run()
    # -------------------------------------------------------------------------
    def run(self):
        """
        Orchestre toutes les etapes du Module M4 en sequence.

        Etapes :
        1. Generation du fichier GAML (calendrier piecewise)
        2. Generation du fichier de configuration DSL (.sir, calendrier)
        3. Generation du rapport JSON complet (regime_calendar)
        4. Affichage des instructions d'integration GAMA

        Returns
        -------
        result : dict -- chemins des fichiers generes
        """
        gaml_path = self.generate_gaml_parameters()
        dsl_path  = self.generate_dsl_config()
        json_path = self.generate_json_report()
        self.show_integration_instructions()

        result = {
            'gaml_file' : gaml_path,
            'dsl_file'  : dsl_path,
            'json_file' : json_path,
        }

        print("\n" + "=" * 65)
        print("  MODULE M4 -- TERMINE AVEC SUCCES")
        print("=" * 65)
        print(f"  Fichiers generes :")
        print(f"  -> {gaml_path}")
        print(f"  -> {dsl_path}")
        print(f"  -> {json_path}")
        print(f"  -> Pret pour Module M5 (validation prospective + sensibilite)")
        print("=" * 65)

        return result


# -----------------------------------------------------------------------------
# POINT D'ENTREE : test autonome
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST AUTONOME -- Module M4 Generation GAML")
    print("=" * 65 + "\n")

    import sys
    sys.path.insert(0, '.')
    from M1_acquisition import AcquisitionModule
    from M2_ia_prior    import IAPriorModule
    from M3_calibration import CalibrationModule

    # M1
    m1 = AcquisitionModule(
        data_path = "data/usa_county_wise.csv",
        state     = "New York",
        N         = 19_450_000,
        D_I       = 14
    )
    df_sir, meta = m1.run(plot=False)

    # M2
    m2           = IAPriorModule(meta=meta, disease="COVID-19")
    prior_result = m2.run(df_sir=df_sir, plot=False)

    # M3
    m3           = CalibrationModule(meta=meta, prior_result=prior_result)
    calib_result = m3.run(df_sir=df_sir, plot=False)

    # M4
    m4     = GAMLGenerationModule(calib_result=calib_result)
    result = m4.run()

    print("\n=== FICHIERS GENERES ===")
    for k, v in result.items():
        print(f"  {k} : {v}")
    print("\n-> Module M4 valide -- pret pour M5 (validation)")
