"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           MODULE M1 — ACQUISITION ET PRÉTRAITEMENT DES DONNÉES             ║
║                                                                              ║
║  Pipeline d'estimation automatique des paramètres SIR (α, β', γ)           ║
║  par Intelligence Artificielle générative                                    ║
║                                                                              ║
║  Auteur  : Henri BIKOURI , henribikouri.github.io                                 ║
║  Dataset : JHU CSSE COVID-19 — USA County-Wise                              ║
║  Maladie : COVID-19 (SARS-CoV-2)                                            ║
║                                                                              ║
║  Rôle de ce module :                                                         ║
║    → Charger les données brutes du dataset CSV                               ║
║    → Agréger par état géographique                                           ║
║    → Calculer les nouveaux cas journaliers ΔI(t)                            ║
║    → Reconstruire les compartiments S(t), I(t), R(t)                        ║
║    → Lisser la série par filtre de Savitzky-Golay                           ║
║    → Exporter une série propre prête pour la calibration (Module M3)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd                        # Manipulation de données tabulaires
import numpy as np                         # Calculs numériques vectorisés
import matplotlib.pyplot as plt            # Visualisation des courbes
import matplotlib.dates as mdates         # Formatage des axes temporels
from scipy.signal import savgol_filter    # Filtre de Savitzky-Golay (lissage)
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH          = "data/usa_county_wise.csv"
TARGET_STATE       = "New York"
N_POPULATION       = 19_450_000   # Population New York (US Census 2020)
D_INFECTIOUS_PRIOR = 14           # Durée infectieuse COVID-19 a priori (OMS)
SG_WINDOW          = 7            # Fenêtre filtre Savitzky-Golay (jours, impair)
SG_POLYORDER       = 3            # Degré du polynôme local SG


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────
class AcquisitionModule:
    """
    Module M1 du pipeline de calibration SIR.

    Responsable de toute la chaîne de traitement des données brutes,
    depuis le fichier CSV jusqu'à la série temporelle propre et structurée
    prête pour l'optimisation numérique (Module M3).

    Attributs
    ---------
    data_path : str        — chemin vers le fichier CSV brut
    state     : str        — nom de l'état à analyser
    N         : int        — taille de la population
    D_I       : int        — durée infectieuse a priori (pour reconstruction)
    df_raw    : DataFrame  — données brutes chargées
    df_state  : DataFrame  — données agrégées pour l'état cible
    df_sir    : DataFrame  — compartiments S, I, R reconstruits et lissés
    """

    def __init__(self, data_path=DATA_PATH, state=TARGET_STATE,
                 N=N_POPULATION, D_I=D_INFECTIOUS_PRIOR):
        self.data_path = data_path
        self.state     = state
        self.N         = N
        self.D_I       = D_I
        self.df_raw    = None
        self.df_state  = None
        self.df_sir    = None

        print("=" * 65)
        print("  MODULE M1 — Acquisition & Prétraitement des données")
        print("=" * 65)
        print(f"  État cible    : {self.state}")
        print(f"  Population N  : {self.N:,} habitants")
        print(f"  Durée inf. D_I: {self.D_I} jours (a priori COVID-19/OMS)")
        print("=" * 65)

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 : Chargement des données brutes
    # ─────────────────────────────────────────────────────────────────────────
    def load_data(self):
        """
        Charge le fichier CSV brut et convertit les types.

        Le dataset JHU CSSE contient des données cumulées par comté
        américain. On charge tout le fichier puis on filtre sur l'état.
        """
        print("\n[M1 - Étape 1] Chargement des données brutes...")

        self.df_raw = pd.read_csv(
            self.data_path,
            parse_dates=['Date'],
            dayfirst=False           # Format américain MM/DD/YY
        )

        # Vérification des colonnes indispensables
        required_cols = ['Date', 'Province_State', 'Confirmed', 'Deaths']
        missing = [c for c in required_cols if c not in self.df_raw.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

        print(f"  -> Dataset chargé : {len(self.df_raw):,} lignes")
        print(f"  -> Période        : {self.df_raw['Date'].min().date()} "
              f"-> {self.df_raw['Date'].max().date()}")
        print(f"  -> États présents : {self.df_raw['Province_State'].nunique()}")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 : Agrégation par état
    # ─────────────────────────────────────────────────────────────────────────
    def aggregate_by_state(self):
        """
        Agrège les données de tous les comtés de l'état cible par date.

        Le dataset JHU est au niveau comté. On somme les cas confirmés
        et décès de tous les comtés de l'état pour obtenir une série
        temporelle agrégée au niveau état.
        """
        print(f"\n[M1 - Étape 2] Agrégation pour : {self.state}...")

        df_filtered = self.df_raw[
            self.df_raw['Province_State'] == self.state
        ].copy()

        if len(df_filtered) == 0:
            raise ValueError(f"Aucune donnée pour l'état : {self.state}")

        # Somme des cas par date (tous comtés confondus)
        self.df_state = df_filtered.groupby('Date').agg(
            Confirmed_cumul=('Confirmed', 'sum'),  # Cas cumulés totaux
            Deaths_cumul   =('Deaths',    'sum')   # Décès cumulés totaux
        ).reset_index().sort_values('Date').reset_index(drop=True)

        print(f"  -> Comtés agrégés    : {df_filtered['Admin2'].nunique()}")
        print(f"  -> Jours disponibles : {len(self.df_state)}")
        print(f"  -> Pic cas cumulés   : {self.df_state['Confirmed_cumul'].max():,}")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 : Calcul des nouveaux cas journaliers
    # ─────────────────────────────────────────────────────────────────────────
    def compute_new_cases(self):
        """
        Calcule les nouveaux cas journaliers deltaI(t) par différentiation.

        Les données JHU sont CUMULÉES.
        On récupère les cas journaliers par :
            deltaI(t) = Confirmed(t) - Confirmed(t-1)

        Des valeurs négatives peuvent apparaître (corrections rétroactives)
        → on les force à 0 (clip lower=0).
        """
        print("\n[M1 - Étape 3] Calcul des nouveaux cas journaliers deltaI(t)...")

        df = self.df_state
        df['new_cases']  = df['Confirmed_cumul'].diff().fillna(0).clip(lower=0)
        df['new_deaths'] = df['Deaths_cumul'].diff().fillna(0).clip(lower=0)

        # Statistiques du pic épidémique
        peak_idx  = df['new_cases'].idxmax()
        peak_date = df.loc[peak_idx, 'Date'].date()
        peak_val  = df.loc[peak_idx, 'new_cases']
        print(f"  -> Pic épidémique     : {peak_val:,.0f} cas/jour le {peak_date}")
        print(f"  -> Moyenne journalière: {df['new_cases'].mean():,.1f} cas/jour")

        self.df_state = df
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 4 : Reconstruction des compartiments S, I, R
    # ─────────────────────────────────────────────────────────────────────────
    def reconstruct_sir_compartments(self):
        """
        Reconstruit S(t), I(t), R(t) à partir des cas confirmés.

        Formules de reconstruction :

          I(t) = somme_glissante(deltaI, D_I jours)
                 Un infecté reste infectieux D_I jours en moyenne.

          R(t) = Confirmed_cumul(t) - I(t)
                 Tous les anciens infectés qui ne sont plus en I.

          S(t) = N - Confirmed_cumul(t)
                 Tous ceux qui n'ont jamais été infectés.

        Note : les décès sont inclus dans R (retirés de la dynamique).
        """
        print(f"\n[M1 - Étape 4] Reconstruction S(t), I(t), R(t)...")
        print(f"  -> Hypothèse D_I = {self.D_I} jours (OMS COVID-19)")

        df = self.df_state

        # I(t) : somme glissante des nouveaux cas sur D_I jours
        df['I_raw'] = (
            df['new_cases']
            .rolling(window=self.D_I, min_periods=1)
            .sum()
        )

        # R(t) : anciens infectés sortis de l'état infectieux
        df['R_raw'] = (df['Confirmed_cumul'] - df['I_raw']).clip(lower=0)

        # S(t) : jamais infectés
        df['S_raw'] = (self.N - df['Confirmed_cumul']).clip(lower=0)

        # Vérification de cohérence : S + I + R doit approcher N
        total = df['S_raw'] + df['I_raw'] + df['R_raw']
        ecart = (total - self.N).abs().max()
        print(f"  -> Verif. S+I+R ~= N : ecart max = {ecart:,.0f} (acceptable)")

        self.df_state = df
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 5 : Lissage par filtre de Savitzky-Golay
    # ─────────────────────────────────────────────────────────────────────────
    def apply_smoothing(self):
        """
        Applique le filtre de Savitzky-Golay pour lisser les séries brutes.

        Pourquoi lisser ?
        Les données brutes présentent des artefacts de reporting :
        - Effets de weekend (moins de tests → faux creux le dimanche)
        - Rattrapages de déclaration (faux pics ponctuels)
        Ces oscillations perturbent l'optimiseur du Module M3.

        Le filtre Savitzky-Golay (1964) ajuste localement un polynôme
        de degré polyorder sur une fenêtre de window_length points.
        Avantage : préserve la position du pic épidémique, contrairement
        à une moyenne mobile simple qui le déplacerait dans le temps.
        """
        print(f"\n[M1 - Étape 5] Lissage Savitzky-Golay "
              f"(fenêtre={SG_WINDOW}j, degré={SG_POLYORDER})...")

        df = self.df_state

        for col_raw, col_smooth in [
            ('I_raw',     'I_smooth'),
            ('S_raw',     'S_smooth'),
            ('R_raw',     'R_smooth'),
            ('new_cases', 'new_cases_smooth')
        ]:
            smoothed = savgol_filter(
                df[col_raw].values,
                window_length=SG_WINDOW,
                polyorder=SG_POLYORDER
            )
            # Clip : le filtre peut produire de légères valeurs négatives
            df[col_smooth] = np.clip(smoothed, 0, None)

        print(f"  -> Lissage appliqué sur S, I, R et nouveaux cas")
        print(f"  -> Effets de weekend atténués")

        self.df_state = df
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 6 : Construction du DataFrame final SIR
    # ─────────────────────────────────────────────────────────────────────────
    def build_sir_dataframe(self):
        """
        Construit le DataFrame final propre — sortie unique du Module M1.

        Ce DataFrame est transmis directement au Module M3 (calibration).

        Colonnes :
          t          : indice temporel en jours (0, 1, 2, ...)
          Date       : date calendaire
          S_data      : susceptibles lissés
          I_data      : infectés actifs lissés
          R_data      : retirés lissés
          S_raw      : susceptibles bruts
          I_raw      : infectés actifs bruts
          R_raw      : retirés bruts
          new_cases  : nouveaux cas journaliers bruts
          new_smooth : nouveaux cas journaliers lissés
        """
        print("\n[M1 - Étape 6] Construction du DataFrame SIR final...")

        df = self.df_state
        self.df_sir = pd.DataFrame({
            't'          : np.arange(len(df)),
            'Date'       : df['Date'].values,
            'S_data'      : df['S_smooth'].values,
            'I_data'      : df['I_smooth'].values,
            'R_data'      : df['R_smooth'].values,
            'S_raw'      : df['S_raw'].values,
            'I_raw'      : df['I_raw'].values,
            'R_raw'      : df['R_raw'].values,
            'new_cases'  : df['new_cases'].values,
            'new_smooth' : df['new_cases_smooth'].values
        })

        T = len(self.df_sir)
        print(f"  -> Série temporelle : {T} jours")
        print(f"  -> I(t) max         : {self.df_sir['I_data'].max():,.0f} infectés actifs")
        print(f"  -> S(t) final       : {self.df_sir['S_data'].iloc[-1]:,.0f} susceptibles restants")
        print(f"  -> R(t) final       : {self.df_sir['R_data'].iloc[-1]:,.0f} retirés")

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALISATION
    # ─────────────────────────────────────────────────────────────────────────
    def plot_data_quality(self, save_path="outputs/M1_data_quality.png"):
        """
        Génère les graphiques de contrôle qualité du prétraitement.

        Graphique 1 : Nouveaux cas bruts vs lissés (vérification lissage)
        Graphique 2 : Compartiments S, I, R reconstruits (vérification SIR)
        """
        os.makedirs("outputs", exist_ok=True)

        df  = self.df_sir
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(
            f"Module M1 — Prétraitement COVID-19 | {self.state} | N={self.N:,}",
            fontsize=14, fontweight='bold'
        )

        # Graphique 1 : Nouveaux cas journaliers bruts vs lissés
        ax1 = axes[0]
        ax1.bar(df['Date'], df['new_cases'],
                color='lightcoral', alpha=0.5, label='Nouveaux cas bruts deltaI(t)')
        ax1.plot(df['Date'], df['new_smooth'],
                 color='darkred', linewidth=2.5,
                 label=f'Lissé Savitzky-Golay (fenêtre={SG_WINDOW}j, deg={SG_POLYORDER})')
        ax1.set_title('Nouveaux cas journaliers — Bruts vs Lissés', fontsize=12)
        ax1.set_ylabel('Nouveaux cas / jour')
        ax1.legend()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.grid(True, alpha=0.3)

        # Annotation du pic
        peak_idx  = df['new_smooth'].idxmax()
        peak_date = df.loc[peak_idx, 'Date']
        peak_val  = df.loc[peak_idx, 'new_smooth']
        ax1.annotate(
            f"Pic: {peak_val:,.0f}\n{peak_date.strftime('%d %b %Y')}",
            xy=(peak_date, peak_val),
            xytext=(peak_date + pd.Timedelta(days=15), peak_val * 0.85),
            arrowprops=dict(arrowstyle='->', color='darkred'),
            fontsize=10, color='darkred', fontweight='bold'
        )

        # Graphique 2 : Compartiments S, I, R
        ax2 = axes[1]
        ax2.plot(df['Date'], df['S_data'] / self.N * 100,
                 color='green',  linewidth=2, label='S(t) Susceptibles (%N)')
        ax2.plot(df['Date'], df['I_data'] / self.N * 100,
                 color='red',    linewidth=2, label='I(t) Infectés actifs (%N)')
        ax2.plot(df['Date'], df['R_data'] / self.N * 100,
                 color='blue',   linewidth=2, label='R(t) Retirés (%N)')
        ax2.set_title('Compartiments SIR reconstruits (% de la population N)', fontsize=12)
        ax2.set_ylabel('Fraction de la population (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  -> Graphique sauvegardé : {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTHODE PRINCIPALE : run()
    # ─────────────────────────────────────────────────────────────────────────
    def run(self, plot=True):
        """
        Orchestre toutes les étapes du Module M1 en séquence.

        Returns
        -------
        df_sir : pd.DataFrame — série temporelle SIR propre
        meta   : dict         — métadonnées pour les modules suivants
        """
        (self
         .load_data()
         .aggregate_by_state()
         .compute_new_cases()
         .reconstruct_sir_compartments()
         .apply_smoothing()
         .build_sir_dataframe()
        )

        if plot:
            self.plot_data_quality()

        # Métadonnées transmises aux modules suivants
        meta = {
            'state'      : self.state,
            'N'          : self.N,
            'D_I_prior'  : self.D_I,
            'T'          : len(self.df_sir),
            'date_start' : self.df_sir['Date'].iloc[0],
            'date_end'   : self.df_sir['Date'].iloc[-1],
            'peak_I'     : self.df_sir['I_data'].max(),
            'disease'    : 'COVID-19',
            't_obs'      : self.df_sir['t'].values,
            'S_data'      : self.df_sir['S_data'].values,
            'I_data'      : self.df_sir['I_data'].values,
            'R_data'      : self.df_sir['R_data'].values,
        }

        print("\n" + "=" * 65)
        print("  MODULE M1 — TERMINE AVEC SUCCES")
        print("=" * 65)
        print(f"  -> Pret pour Module M2 (extraction prior par IA)")
        print(f"  -> {meta['T']} points | Pic I = {meta['peak_I']:,.0f} infectés actifs")
        print("=" * 65)

        return self.df_sir, meta


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE : test autonome
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST AUTONOME — Module M1 Acquisition")
    print("=" * 65 + "\n")

    m1 = AcquisitionModule(
        data_path="data/usa_county_wise.csv",
        state="New York",
        N=19_450_000,
        D_I=14
    )

    df_sir, meta = m1.run(plot=True)

    print("\n=== APERCU DU DATAFRAME SIR FINAL ===")
    print(df_sir[['t', 'Date', 'S_data', 'I_data', 'R_data',
                   'new_cases', 'new_smooth']].head(10).to_string(index=False))
    print("...")
    print(df_sir[['t', 'Date', 'S_data', 'I_data', 'R_data',
                   'new_cases', 'new_smooth']].tail(5).to_string(index=False))

    print("\n=== METADONNEES ===")
    for k, v in meta.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k:15s} : {v}")

    print("\n-> Module M1 valide — pret pour M2")
