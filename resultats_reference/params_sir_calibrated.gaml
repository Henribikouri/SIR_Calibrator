/**
 * ==========================================================================
 * CALENDRIER DE PARAMETRES SIR theta(t) -- GENERE AUTOMATIQUEMENT
 * ==========================================================================
 *
 * Pipeline : Estimation automatique sequentielle des parametres SIR par IA
 *            generative (calibration par fenetres glissantes chainees,
 *            population COMPLETE -- pas de pool effectif)
 * Auteur   : Henri (Master 2 Genie Logiciel)
 * Date     : 2026-08-07 11:14:43
 *
 * Maladie  : Ebola (EVD)
 * Region   : Sierra Leone
 * Pop. N   : 5,743,725 habitants (population COMPLETE)
 *
 * Methode de calibration :
 *   52 fenetres theta_k, chacune calibree par Evolution Differentielle
 *   (exploration globale) puis Nelder-Mead (raffinement local), avec
 *   continuite exacte de l'etat (S,I,R) entre fenetres successives.
 *
 * Calendrier de regimes (jour de bascule -> alpha_k, gamma_k, R0_k) :
 *   regime  0 : jours   0-4   | alpha=0.3594  gamma=0.3873  R0=0.928
 *   regime  1 : jours   5-9   | alpha=0.2396  gamma=0.1307  R0=1.833
 *   regime  2 : jours  10-14  | alpha=0.2429  gamma=0.1197  R0=2.029
 *   regime  3 : jours  15-19  | alpha=0.2050  gamma=0.1337  R0=1.533
 *   regime  4 : jours  20-24  | alpha=0.1878  gamma=0.1565  R0=1.200
 *   regime  5 : jours  25-29  | alpha=0.1895  gamma=0.1839  R0=1.030
 *   regime  6 : jours  30-34  | alpha=0.1612  gamma=0.2411  R0=0.669
 *   regime  7 : jours  35-39  | alpha=0.1392  gamma=0.3244  R0=0.429
 *   regime  8 : jours  40-44  | alpha=0.1254  gamma=0.2229  R0=0.563
 *   regime  9 : jours  45-49  | alpha=0.1298  gamma=0.2376  R0=0.546
 *   regime 10 : jours  50-54  | alpha=0.1489  gamma=0.1889  R0=0.788
 *   regime 11 : jours  55-59  | alpha=0.1692  gamma=0.1726  R0=0.981
 *   regime 12 : jours  60-64  | alpha=0.1809  gamma=0.1681  R0=1.076
 *   regime 13 : jours  65-69  | alpha=0.1782  gamma=0.1975  R0=0.902
 *   regime 14 : jours  70-74  | alpha=0.1799  gamma=0.1768  R0=1.018
 *   regime 15 : jours  75-79  | alpha=0.1888  gamma=0.1736  R0=1.087
 *   regime 16 : jours  80-84  | alpha=0.1789  gamma=0.2138  R0=0.837
 *   regime 17 : jours  85-89  | alpha=0.1632  gamma=0.2106  R0=0.775
 *   regime 18 : jours  90-94  | alpha=0.1921  gamma=0.1296  R0=1.482
 *   regime 19 : jours  95-99  | alpha=0.1778  gamma=0.2070  R0=0.859
 *   regime 20 : jours 100-104 | alpha=0.1792  gamma=0.3750  R0=0.478
 *   regime 21 : jours 105-109 | alpha=0.1750  gamma=0.1457  R0=1.202
 *   regime 22 : jours 110-114 | alpha=0.1759  gamma=0.1910  R0=0.921
 *   regime 23 : jours 115-119 | alpha=0.1800  gamma=0.1685  R0=1.068
 *   regime 24 : jours 120-124 | alpha=0.1914  gamma=0.1783  R0=1.074
 *   regime 25 : jours 125-129 | alpha=0.1893  gamma=0.1792  R0=1.057
 *   regime 26 : jours 130-134 | alpha=0.1840  gamma=0.1960  R0=0.939
 *   regime 27 : jours 135-139 | alpha=0.1746  gamma=0.2212  R0=0.789
 *   regime 28 : jours 140-144 | alpha=0.1794  gamma=0.1686  R0=1.064
 *   regime 29 : jours 145-149 | alpha=0.1766  gamma=0.1972  R0=0.896
 *   regime 30 : jours 150-154 | alpha=0.1608  gamma=0.2521  R0=0.638
 *   regime 31 : jours 155-159 | alpha=0.1633  gamma=0.1991  R0=0.820
 *   regime 32 : jours 160-164 | alpha=0.1756  gamma=0.1769  R0=0.993
 *   regime 33 : jours 165-169 | alpha=0.2021  gamma=0.1442  R0=1.402
 *   regime 34 : jours 170-174 | alpha=0.2131  gamma=0.1377  R0=1.547
 *   regime 35 : jours 175-179 | alpha=0.1919  gamma=0.1984  R0=0.967
 *   regime 36 : jours 180-184 | alpha=0.1764  gamma=0.2755  R0=0.640
 *   regime 37 : jours 185-189 | alpha=0.1655  gamma=0.1808  R0=0.916
 *   regime 38 : jours 190-194 | alpha=0.1859  gamma=0.1389  R0=1.338
 *   regime 39 : jours 195-199 | alpha=0.1952  gamma=0.1663  R0=1.174
 *   regime 40 : jours 200-204 | alpha=0.1987  gamma=0.1758  R0=1.130
 *   regime 41 : jours 205-209 | alpha=0.1957  gamma=0.1853  R0=1.056
 *   regime 42 : jours 210-214 | alpha=0.1920  gamma=0.1815  R0=1.058
 *   regime 43 : jours 215-219 | alpha=0.1757  gamma=0.2163  R0=0.812
 *   regime 44 : jours 220-224 | alpha=0.1636  gamma=0.2463  R0=0.664
 *   regime 45 : jours 225-229 | alpha=0.1849  gamma=0.1493  R0=1.238
 *   regime 46 : jours 230-234 | alpha=0.1905  gamma=0.1830  R0=1.041
 *   regime 47 : jours 235-239 | alpha=0.1675  gamma=0.2509  R0=0.667
 *   regime 48 : jours 240-244 | alpha=0.0820  gamma=0.3893  R0=0.211
 *   regime 49 : jours 245-249 | alpha=0.0820  gamma=0.3893  R0=0.211
 *   regime 50 : jours 250-254 | alpha=0.0820  gamma=0.3893  R0=0.211
 *   regime 51 : jours 255-257 | alpha=0.0820  gamma=0.3893  R0=0.211
 *
 * Metriques de validation (reconstruction globale, 5,743,725 hab.) :
 *   RMSE = 129.1 cas/jour
 *   MAE  = 55.9 cas/jour
 *   R2   = 0.9553
 *   Qualite : EXCELLENT
 *   Erreur pic = 1 jours
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

global {

    // -----------------------------------------------------------------------
    // CALENDRIER DE REGIMES theta(t) -- listes paralleles
    // regime_start_day[i] = jour (cycle GAMA) a partir duquel le regime i
    // s'applique. regime_infection_rate[i] / regime_recovery_rate[i] = les
    // valeurs (alpha_i, gamma_i) calibrees sur la fenetre i.
    // -----------------------------------------------------------------------
    list<int>   regime_start_day      <- [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255];
    list<float> regime_infection_rate <- [0.359408, 0.239624, 0.242926, 0.205036, 0.187849, 0.189492, 0.161237, 0.139166, 0.125423, 0.129765, 0.148887, 0.169247, 0.180881, 0.178176, 0.179929, 0.188784, 0.178885, 0.163183, 0.192129, 0.177774, 0.179169, 0.175043, 0.175894, 0.180042, 0.191438, 0.189296, 0.184023, 0.174606, 0.179367, 0.176644, 0.160836, 0.163252, 0.175581, 0.202142, 0.213136, 0.191898, 0.176381, 0.165535, 0.185865, 0.195160, 0.198662, 0.195660, 0.192001, 0.175691, 0.163602, 0.184861, 0.190503, 0.167465, 0.082004, 0.082004, 0.082004, 0.082004];
    list<float> regime_recovery_rate  <- [0.387272, 0.130723, 0.119727, 0.133724, 0.156505, 0.183893, 0.241115, 0.324419, 0.222870, 0.237570, 0.188913, 0.172564, 0.168105, 0.197490, 0.176795, 0.173590, 0.213815, 0.210590, 0.129611, 0.207006, 0.374976, 0.145668, 0.190954, 0.168545, 0.178257, 0.179152, 0.196042, 0.221155, 0.168631, 0.197233, 0.252139, 0.199088, 0.176857, 0.144155, 0.137744, 0.198364, 0.275490, 0.180752, 0.138945, 0.166263, 0.175751, 0.185282, 0.181486, 0.216267, 0.246341, 0.149315, 0.183000, 0.250902, 0.389327, 0.389327, 0.389327, 0.389327];
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
     * Prior IA : 7 ticks | Fixe : 7 ticks
     */
    int min_infectious_time <- 7;

    // -----------------------------------------------------------------------
    // BASCULE DE REGIME -- execute a chaque cycle, change infection_rate et
    // recovery_rate des que le cycle courant atteint le jour de bascule
    // suivant. Reproduit fidelement la trajectoire theta(t) calibree.
    // -----------------------------------------------------------------------
    reflex maj_regime_theta when: cycle > 0 {
        int idx <- current_regime_index;
        loop i from: 0 to: length(regime_start_day) - 1 {
            if (cycle >= regime_start_day[i]) {
                idx <- i;
            }
        }
        if (idx != current_regime_index) {
            current_regime_index <- idx;
            infection_rate <- regime_infection_rate[idx];
            recovery_rate  <- regime_recovery_rate[idx];
            write "Bascule regime theta_" + idx + " (jour " + cycle + ") -> "
                + "alpha=" + infection_rate + " gamma=" + recovery_rate;
        }
    }

    // -----------------------------------------------------------------------
    // PARAMETRES DERIVES (dernier regime -- reference / affichage)
    // -----------------------------------------------------------------------

    /**
     * R0* = Nombre de reproduction de base du DERNIER regime calibre
     * R0 = alpha_K / gamma_K = 0.082004 / 0.389327
     * R0 > 1 : epidemie se propage | R0 < 1 : epidemie s'eteint
     * NOTE : R0(t) varie regime par regime -- voir regime_infection_rate /
     * regime_recovery_rate ci-dessus pour la trajectoire complete.
     */
    float R0_calibrated <- 0.210600;

    /**
     * Duree moyenne de la maladie (theorique, dernier regime)
     * E[duree] = beta'* + (1 - gamma_K) / gamma_K
     *          = 7 + (1 - 0.3893) / 0.3893
     *          = 8.6 jours
     */
    float mean_disease_duration <- 8.6000;

    // -----------------------------------------------------------------------
    // METADONNEES DE CALIBRATION (pour tracabilite et reproductibilite)
    // -----------------------------------------------------------------------
    string calibration_method  <- "Calibration sequentielle theta(t) : "
                                 + "fenetres glissantes chainees, DE + Nelder-Mead";
    string calibration_disease <- "Ebola (EVD)";
    string calibration_region  <- "Sierra Leone";
    int    calibration_n_regimes <- 52;
    float  calibration_RMSE    <- 129.12;
    float  calibration_R2      <- 0.955300;
    string calibration_date    <- "2026-08-07 11:14:43";

}
