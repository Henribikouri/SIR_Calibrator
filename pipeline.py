"""
=============================================================================
  PIPELINE.PY -- ORCHESTRATEUR PRINCIPAL
=============================================================================

  Pipeline d'estimation automatique des parametres SIR (alpha, beta', gamma)
  par Intelligence Artificielle generative

  Auteur  : Henri BIKOURI

  Ce script orchestre les 5 modules du pipeline en une seule commande :

      python pipeline.py

  Flux d'execution :
      M1 (Acquisition)
        --> M2 (Prior IA)
          --> M3 (Calibration PSO + Nelder-Mead)
            --> M4 (Generation GAML)
              --> M5 (Validation + Sobol)
                --> Rapport final + Fichiers GAMA

  Arguments en ligne de commande (optionnels) :
      --data    : chemin du fichier CSV (defaut: data/usa_county_wise.csv)
      --state   : etat/region cible    (defaut: New York)
      --N       : population totale    (defaut: 19450000)
      --disease : nom de la maladie    (defaut: COVID-19)
      --mode    : format CSV           (defaut: auto)
      --sobol   : echantillons Sobol   (defaut: 256)
      --no-plot : desactiver graphiques

  Exemples d'utilisation :

      # Cas 1 : Dataset JHU New York (par defaut)
      python pipeline.py

      # Cas 2 : CSV de sortie GAMA camerounais
      python pipeline.py --data data/sir_output_cameroun.csv
                         --state Cameroun --N 5000 --mode gama

      # Cas 3 : Autre dataset epidemio
      python pipeline.py --data data/mon_dataset.csv
                         --state "Ma Region" --N 1000000
                         --disease "Grippe"

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import argparse         # Gestion des arguments en ligne de commande
import sys              # Acces au systeme
import os               # Gestion des fichiers
import time             # Mesure du temps d'execution
import json             # Serialisation des resultats
import traceback        # Affichage des erreurs detaillees
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------
# FONCTION PRINCIPALE : run_pipeline()
# -----------------------------------------------------------------------------
def run_pipeline(
    data_path = "data/usa_county_wise.csv",
    state     = "New York",
    N         = 19_450_000,
    disease   = "COVID-19",
    mode      = "auto",
    D_I       = 14,
    n_sobol   = 256,
    plot      = True
):
    """
    Orchestre l'execution complete du pipeline M1 -> M5.

    C'est la fonction principale du projet. Elle appelle
    sequentiellement chaque module et transmet les resultats
    au module suivant.

    Parameters
    ----------
    data_path : str  -- chemin du fichier CSV de donnees
    state     : str  -- zone geographique cible
    N         : int  -- population totale
    disease   : str  -- nom de la maladie
    mode      : str  -- format CSV ("auto", "jhu", "gama", "generic")
    D_I       : int  -- duree infectieuse a priori (jours)
    n_sobol   : int  -- echantillons pour analyse de Sobol
    plot      : bool -- generer les graphiques

    Returns
    -------
    results : dict -- tous les resultats du pipeline
    """

    t_start = time.time()
    # Prefixe personnalise pour tous les fichiers de sortie
    # Format : NomRegion_Population  ex: NewYork_19450000
    region_clean = state.replace(" ", "_").replace("'", "")
    prefix       = f"{region_clean}_{N}"


    # =========================================================================
    # BANNIERE D'INTRODUCTION
    # =========================================================================
    print("\n")
    print("=" * 65)
    print("  PIPELINE SIR -- ESTIMATION AUTOMATIQUE DES PARAMETRES")
    print("  par Intelligence Artificielle Generative")
    print("=" * 65)
    print(f"  Maladie    : {disease}")
    print(f"  Region     : {state}")
    print(f"  Population : {N:,} habitants")
    print(f"  Donnees    : {data_path}")
    print(f"  Mode CSV   : {mode}")
    print("=" * 65)
    print()

    # =========================================================================
    # CHARGEMENT DES MODULES
    # =========================================================================
    print("[PIPELINE] Chargement des modules...")
    sys.path.insert(0, '.')

    # Chargement de M1 -- version universelle si disponible
    try:
        from M1_acquisition_cameroun import AcquisitionModuleUniversel as M1
        print("  -> M1 : AcquisitionModuleUniversel (mode universel)")
        use_universal = True
    except ImportError:
        from M1_acquisition import AcquisitionModule as M1
        print("  -> M1 : AcquisitionModule (mode JHU)")
        use_universal = False

    from M2_ia_prior    import IAPriorModule
    from M3_calibration import CalibrationModule
    from M4_generation_gaml import GAMLGenerationModule
    from M5_validation  import ValidationModule

    print("  -> M2 : IAPriorModule       (prior IA Gemini/Claude)")
    print("  -> M3 : CalibrationModule   (theta(t) sequentiel, DE + Nelder-Mead)")
    print("  -> M4 : GAMLGenerationModule (calendrier GAML piecewise)")
    print("  -> M5 : ValidationModule    (Sobol + validation prospective)")
    print()

    # =========================================================================
    # MODULE M1 -- ACQUISITION ET PRETRAITEMENT
    # =========================================================================
    print("=" * 65)
    print("  ETAPE 1/5 -- MODULE M1 : Acquisition des donnees")
    print("=" * 65)
    t1 = time.time()

    if use_universal:
        m1 = M1(
            data_path = data_path,
            N         = N,
            mode      = mode,
            state     = state,
            D_I       = D_I
        )
    else:
        m1 = M1(
            data_path = data_path,
            state     = state,
            N         = N,
            D_I       = D_I
        )

    df_sir, meta = m1.run(
        plot      = plot,
        save_path = f"outputs/M1_{prefix}.png"
    )
    meta['disease'] = disease

    print(f"\n  [M1 OK] -- {time.time()-t1:.1f}s")
    print(f"  -> {meta['T']} jours | Pic I = {meta['peak_I']:,.0f}")

    # =========================================================================
    # MODULE M2 -- EXTRACTION DES PRIOR PAR IA
    # =========================================================================
    print("\n" + "=" * 65)
    print("  ETAPE 2/5 -- MODULE M2 : Extraction Prior par IA")
    print("=" * 65)
    t2 = time.time()

    m2           = IAPriorModule(meta=meta, disease=disease)
    prior_result = m2.run(
        df_sir    = df_sir,
        plot      = plot,
        save_path = f"outputs/M2_{prefix}.png"
    )

    print(f"\n  [M2 OK] -- {time.time()-t2:.1f}s")
    print(f"  -> theta_0 = (alpha={prior_result['alpha_0']}, "
          f"beta'={prior_result['beta_prime_0']}, "
          f"gamma={prior_result['gamma_0']})")
    print(f"  -> R0 prior = {prior_result['R0_prior']}")

    # =========================================================================
    # MODULE M3 -- CALIBRATION NUMERIQUE
    # =========================================================================
    print("\n" + "=" * 65)
    print("  ETAPE 3/5 -- MODULE M3 : Calibration sequentielle theta(t)")
    print("=" * 65)
    t3 = time.time()

    m3           = CalibrationModule(meta=meta, prior_result=prior_result)
    calib_result = m3.run(
        df_sir    = df_sir,
        plot      = plot,
        save_path = f"outputs/M3_{prefix}.png"
    )

    print(f"\n  [M3 OK] -- {time.time()-t3:.1f}s")
    print(f"  -> {calib_result['n_windows']} fenetres theta_k (dernier regime affiche ci-dessous)")
    print(f"  -> alpha* = {calib_result['alpha_star']:.6f}")
    print(f"  -> beta'* = {calib_result['beta_prime_star']} jours")
    print(f"  -> gamma* = {calib_result['gamma_star']:.6f}")
    print(f"  -> R0*    = {calib_result['R0_star']:.4f}")
    print(f"  -> RMSE   = {calib_result['RMSE']:.1f} cas/jour  (serie complete, N entier)")
    print(f"  -> R2     = {calib_result['R2']:.4f}")

    # =========================================================================
    # MODULE M4 -- GENERATION DU FICHIER GAML
    # =========================================================================
    print("\n" + "=" * 65)
    print("  ETAPE 4/5 -- MODULE M4 : Generation fichier GAMA")
    print("=" * 65)
    t4 = time.time()

    m4     = GAMLGenerationModule(calib_result=calib_result)
    m4_result = m4.run()

    print(f"\n  [M4 OK] -- {time.time()-t4:.1f}s")
    print(f"  -> {m4_result['gaml_file']}")
    print(f"  -> {m4_result['dsl_file']}")
    print(f"  -> {m4_result['json_file']}")

    # =========================================================================
    # MODULE M5 -- VALIDATION ET ANALYSE DE SENSIBILITE
    # =========================================================================
    print("\n" + "=" * 65)
    print("  ETAPE 5/5 -- MODULE M5 : Validation + Analyse Sobol")
    print("=" * 65)
    t5 = time.time()

    m5 = ValidationModule(
        meta         = meta,
        prior_result = prior_result,
        calib_result = calib_result
    )
    val_result = m5.run(
        df_sir    = df_sir,
        plot      = plot,
        n_sobol   = n_sobol,
        save_path = f"outputs/M5_{prefix}.png"
    )

    print(f"\n  [M5 OK] -- {time.time()-t5:.1f}s")

    # =========================================================================
    # RAPPORT FINAL DE SYNTHESE
    # =========================================================================
    t_total = time.time() - t_start

    print("\n")
    print("*" * 65)
    print("  PIPELINE COMPLET -- RESULTATS FINAUX")
    print("*" * 65)
    print()
    print(f"  MALADIE    : {disease} | REGION : {state}")
    print(f"  POPULATION : {N:,} habitants")
    print()
    print(f"  PARAMETRES SIR CALIBRES (a injecter dans GAMA) :")
    print(f"  +--------------------------+------------+------------+")
    print(f"  | Parametre                | Prior IA   | Calibre    |")
    print(f"  +--------------------------+------------+------------+")
    print(f"  | infection_rate (alpha)   | "
          f"{prior_result['alpha_0']:<10.4f} | "
          f"{calib_result['alpha_star']:<10.6f} |")
    print(f"  | min_infectious_time (b') | "
          f"{prior_result['beta_prime_0']:<10} | "
          f"{calib_result['beta_prime_star']:<10} |")
    print(f"  | recovery_rate (gamma)    | "
          f"{prior_result['gamma_0']:<10.4f} | "
          f"{calib_result['gamma_star']:<10.6f} |")
    print(f"  | R0 = alpha/gamma         | "
          f"{prior_result['R0_prior']:<10.3f} | "
          f"{calib_result['R0_star']:<10.4f} |")
    print(f"  +--------------------------+------------+------------+")
    print(f"  | Population calibree : N COMPLET (pas de pool effectif)     |")
    print(f"  |   -> {calib_result['n_windows']} fenetres theta_k sur {meta['T']} jours "
          f"(pas {calib_result['window_step_days']}j / contexte "
          f"{calib_result['window_size_days']}j)")
    print()
    print(f"  METRIQUES DE VALIDATION :")
    print(f"  -> RMSE reconstruction : {calib_result['RMSE']:.1f} cas/jour")
    print(f"  -> R2   reconstruction : {calib_result['R2']:.4f}")
    pr = val_result['prospective_result']
    if pr:
        print(f"  -> RMSE prospectif (one-step-ahead) : {pr['rmse_prospectif']:.1f} cas/jour")
        print(f"  -> R2   prospectif (one-step-ahead) : {pr['r2_prospectif']:.4f}")
    sr = val_result['sobol_result']
    if sr:
        dominant = max(sr['S1'], key=sr['S1'].get)
        print(f"  -> Sobol dominant   : {dominant} "
              f"(S1={sr['S1'][dominant]:.4f})")
    print()
    print(f"  FICHIERS GENERES :")
    print(f"  -> {m4_result['gaml_file']}  (injection GAMA)")
    print(f"  -> {m4_result['dsl_file']}   (DSL config)")
    print(f"  -> {m4_result['json_file']}  (rapport M3)")
    print(f"  -> outputs/rapport_final_pipeline.json (rapport M1-M5)")
    print(f"  -> outputs/M1_{prefix}.png")
    print(f"  -> outputs/M2_{prefix}.png")
    print(f"  -> outputs/M3_{prefix}.png")
    print(f"  -> outputs/M5_{prefix}.png")
    print()
    print(f"  TEMPS TOTAL : {t_total:.1f}s")
    print()
    print(f"  INJECTION GAMA -- copiez ces lignes dans SIRMobiliteflux.gaml :")
    print(f"  -----------------------------------------------------------")
    print(f"  float infection_rate      <- "
          f"{calib_result['alpha_star']:.6f};")
    print(f"  int   min_infectious_time <- "
          f"{calib_result['beta_prime_star']};")
    print(f"  float recovery_rate       <- "
          f"{calib_result['gamma_star']:.6f};")
    print(f"  -----------------------------------------------------------")
    print()
    print("*" * 65)
    print("  PIPELINE TERMINE AVEC SUCCES !")
    print("*" * 65)

    # Dictionnaire de retour complet
    results = {
        'meta'         : {k: v for k, v in meta.items()
                          if k not in ['t_obs','S_data','I_data','R_data']},
        'prior'        : prior_result,
        'calibration'  : calib_result,
        'generation'   : m4_result,
        'validation'   : val_result,
        'temps_total'  : round(t_total, 1)
    }

    return results


# -----------------------------------------------------------------------------
# GESTION DES ARGUMENTS EN LIGNE DE COMMANDE
# -----------------------------------------------------------------------------
def parse_arguments():
    """
    Parse les arguments en ligne de commande.

    Permet d'utiliser le pipeline avec n'importe quel dataset
    sans modifier le code source.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline d'estimation automatique des parametres SIR "
            "par IA generative\n"
            "Auteur : Henri (Master 2 Genie Logiciel)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Dataset JHU New York (par defaut)
  python pipeline.py

  # CSV de sortie GAMA camerounais
  python pipeline.py --data data/sir_output_cameroun.csv
                     --state Cameroun --N 5000 --mode gama

  # Autre dataset avec maladie differente
  python pipeline.py --data data/grippe.csv
                     --state "Ile-de-France" --N 12000000
                     --disease "Grippe H3N2"
        """
    )

    parser.add_argument(
        '--data',
        type    = str,
        default = "data/usa_county_wise.csv",
        help    = "Chemin du fichier CSV de donnees epidemiologiques"
    )
    parser.add_argument(
        '--state',
        type    = str,
        default = "New York",
        help    = "Nom de la zone geographique cible"
    )
    parser.add_argument(
        '--N',
        type    = int,
        default = 19_450_000,
        help    = "Taille de la population totale"
    )
    parser.add_argument(
        '--disease',
        type    = str,
        default = "COVID-19",
        help    = "Nom de la maladie modelisee"
    )
    parser.add_argument(
        '--mode',
        type    = str,
        default = "auto",
        choices = ["auto", "jhu", "gama", "generic"],
        help    = "Format du CSV (auto=detection automatique)"
    )
    parser.add_argument(
        '--D-I',
        type    = int,
        default = 14,
        dest    = 'D_I',
        help    = "Duree infectieuse a priori en jours (defaut: 14)"
    )
    parser.add_argument(
        '--sobol',
        type    = int,
        default = 256,
        help    = "Nombre d'echantillons pour l'analyse de Sobol"
    )
    parser.add_argument(
        '--no-plot',
        action  = 'store_true',
        help    = "Desactiver la generation des graphiques"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# POINT D'ENTREE PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    args = parse_arguments()

    try:
        results = run_pipeline(
            data_path = args.data,
            state     = args.state,
            N         = args.N,
            disease   = args.disease,
            mode      = args.mode,
            D_I       = args.D_I,
            n_sobol   = args.sobol,
            plot      = not args.no_plot
        )

    except KeyboardInterrupt:
        print("\n\n!! Pipeline interrompu par l'utilisateur")
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n!! Fichier introuvable : {e}")
        print(f"   Verifiez que le fichier CSV existe dans le dossier data/")
        sys.exit(1)

    except Exception as e:
        print(f"\n!! Erreur inattendue : {e}")
        print(f"   Details :")
        traceback.print_exc()
        sys.exit(1)