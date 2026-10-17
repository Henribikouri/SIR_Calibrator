"""
=============================================================================
  MODULE M1 -- VERSION UNIVERSELLE (adapte au CSV de sortie GAMA)
=============================================================================

  Ce module adapte M1 pour lire directement le fichier CSV
  genere par la simulation GAMA camerounaise.

  Formats acceptes :
    Format 1 -- CSV JHU (usa_county_wise.csv) : colonnes Date, Confirmed
    Format 2 -- CSV GAMA (sir_output_cameroun.csv) : colonnes tick, S, I, R
    Format 3 -- CSV generique : detection automatique des colonnes

  Auteur : Henri Bikouri

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
import os
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONSTANTES
# -----------------------------------------------------------------------------
SG_WINDOW    = 7
SG_POLYORDER = 3


class AcquisitionModuleUniversel:
    """
    Version universelle de M1 -- accepte n'importe quel CSV epidemiologique.

    Detecte automatiquement le format du CSV et reconstruit
    les compartiments S(t), I(t), R(t) dans tous les cas.

    Modes supportes :
    - mode="jhu"     : format JHU USA (usa_county_wise.csv)
    - mode="gama"    : format sortie GAMA (tick, S_total, I_total, R_total)
    - mode="auto"    : detection automatique (recommande)
    """

    def __init__(self, data_path, N, mode="auto",
                 state="Cameroun", D_I=14,
                 col_date=None, col_cases=None,
                 col_S=None, col_I=None, col_R=None):
        """
        Parameters
        ----------
        data_path : str  -- chemin vers le fichier CSV
        N         : int  -- population totale de la zone
        mode      : str  -- "jhu", "gama", ou "auto"
        state     : str  -- nom de la zone geographique
        D_I       : int  -- duree infectieuse a priori (si reconstruction S/I/R)
        col_date  : str  -- nom colonne date (si mode="auto")
        col_cases : str  -- nom colonne cas confirmes (si mode="auto")
        col_S     : str  -- nom colonne S (si deja disponible)
        col_I     : str  -- nom colonne I (si deja disponible)
        col_R     : str  -- nom colonne R (si deja disponible)
        """
        self.data_path = data_path
        self.N         = N
        self.mode      = mode
        self.state     = state
        self.D_I       = D_I
        self.col_date  = col_date
        self.col_cases = col_cases
        self.col_S     = col_S
        self.col_I     = col_I
        self.col_R     = col_R
        self.df_sir    = None

        print("=" * 65)
        print("  MODULE M1 UNIVERSEL -- Acquisition des donnees")
        print("=" * 65)
        print(f"  Fichier       : {data_path}")
        print(f"  Zone          : {state}")
        print(f"  Population N  : {N:,}")
        print(f"  Mode          : {mode}")
        print("=" * 65)

    # -------------------------------------------------------------------------
    # Detection automatique du format
    # -------------------------------------------------------------------------
    def _detect_format(self, df):
        """
        Detecte automatiquement le format du CSV.

        Regles de detection :
        - Si colonnes "tick" + "S_total"/"I_total" --> format GAMA--- ceci est issu du 
        - Si colonnes "Date" + "Confirmed"          --> format JHU
        - Si colonnes S/I/R fournies manuellement   --> format custom
        - Sinon                                     --> format generique
        """
        cols = [c.lower() for c in df.columns]

        # Format GAMA : tick, S_total, I_total, R_total
        if 'tick' in cols and any('i_total' in c or 'i total' in c
                                   for c in cols):
            print("  -> Format detecte : GAMA (tick, S, I, R)")
            return "gama"

        # Format JHU : Date, Confirmed
        if 'date' in cols and 'confirmed' in cols:
            print("  -> Format detecte : JHU (Date, Confirmed)")
            return "jhu"

        # Format avec S, I, R directement
        sir_cols = ['s', 'i', 'r', 'susceptible', 'infected',
                    'infectious', 'recovered', 'removed']
        if sum(1 for c in cols if any(s in c for s in sir_cols)) >= 2:
            print("  -> Format detecte : SIR direct")
            return "sir_direct"

        print("  -> Format detecte : generique (cas confirmes)")
        return "generic"

    # -------------------------------------------------------------------------
    # Lecture format GAMA
    # -------------------------------------------------------------------------
    def _read_gama_format(self, df):
        """
        Lit le CSV de sortie GAMA avec colonnes :
        tick, S_total, I_total, R_total

        Le 'tick' GAMA est converti en indice temporel entier.
        Les colonnes S, I, R sont lues directement -- pas de reconstruction.
        """
        print("  -> Lecture format GAMA...")

        # Detection flexible des noms de colonnes
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if 'tick' in cl or 'cycle' in cl or 'step' in cl:
                col_map['tick'] = col
            elif cl in ['s_total', 's total', 's', 'susceptible',
                        'sains', 'sain']:
                col_map['S'] = col
            elif cl in ['i_total', 'i total', 'i', 'infected',
                        'infecte', 'infectes', 'infectious']:
                col_map['I'] = col
            elif cl in ['r_total', 'r total', 'r', 'recovered',
                        'retabli', 'retablis', 'removed']:
                col_map['R'] = col

        # Verification des colonnes trouvees
        missing = [k for k in ['tick', 'I'] if k not in col_map]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans le CSV GAMA : {missing}\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )

        # Construction du DataFrame SIR
        t = df[col_map['tick']].values.astype(int)
        I = df[col_map['I']].values.astype(float)

        # S et R optionnels -- reconstruits si absents
        if 'S' in col_map:
            S = df[col_map['S']].values.astype(float)
        else:
            S = np.maximum(0, self.N - I - (df[col_map.get('R',
                           col_map['I'])].values.astype(float)
                           if 'R' in col_map else 0))

        if 'R' in col_map:
            R = df[col_map['R']].values.astype(float)
        else:
            R = np.maximum(0, self.N - S - I)

        # Nouveaux cas = diff de (N - S)
        new_cases = np.maximum(0, np.diff(self.N - S,
                                           prepend=self.N - S[0]))

        print(f"  -> {len(t)} ticks lus")
        print(f"  -> Pic I(t)    : {I.max():,.0f} infectes")
        print(f"  -> S final     : {S[-1]:,.0f} susceptibles")
        print(f"  -> R final     : {R[-1]:,.0f} retablis")

        return t, S, I, R, new_cases

    # -------------------------------------------------------------------------
    # Lecture format JHU
    # -------------------------------------------------------------------------
    def _read_jhu_format(self, df, state_filter=None):
        """
        Lit le CSV au format JHU (Date, Province_State, Confirmed).
        Filtre optionnellement sur un etat/region.
        """
        print("  -> Lecture format JHU...")

        df['Date'] = pd.to_datetime(df['Date'])

        if state_filter and 'Province_State' in df.columns:
            df = df[df['Province_State'] == state_filter]

        # Agregation par date si plusieurs lignes par date
        if 'Province_State' in df.columns:
            df = df.groupby('Date')['Confirmed'].sum().reset_index()
        else:
            df = df.sort_values('Date')

        df['new_cases'] = df['Confirmed'].diff().fillna(0).clip(lower=0)

        t = np.arange(len(df))
        I = (df['new_cases']
             .rolling(window=self.D_I, min_periods=1)
             .sum().values)
        S = np.maximum(0, self.N - df['Confirmed'].values)
        R = np.maximum(0, self.N - S - I)

        print(f"  -> {len(t)} jours lus")
        print(f"  -> Pic I(t) : {I.max():,.0f}")

        return t, S, I, R, df['new_cases'].values

    # -------------------------------------------------------------------------
    # Lecture format generique
    # -------------------------------------------------------------------------
    def _read_generic_format(self, df):
        """
        Lit un CSV generique en detectant automatiquement
        la colonne de cas et la colonne temporelle.
        """
        print("  -> Lecture format generique...")

        # Detection colonne temporelle
        t_col = self.col_date
        if t_col is None:
            for c in df.columns:
                if any(k in c.lower() for k in ['date', 'day', 'jour',
                                                  'week', 'time', 'tick']):
                    t_col = c
                    break

        # Detection colonne cas
        i_col = self.col_cases or self.col_I
        if i_col is None:
            for c in df.columns:
                if any(k in c.lower() for k in ['case', 'cas', 'confirm',
                                                  'infect', 'positive']):
                    i_col = c
                    break

        if i_col is None:
            raise ValueError(
                f"Impossible de detecter la colonne de cas.\n"
                f"Colonnes disponibles : {list(df.columns)}\n"
                f"Specifiez col_cases='nom_colonne' a l'initialisation."
            )

        print(f"  -> Colonne temps    : {t_col}")
        print(f"  -> Colonne cas      : {i_col}")

        # Reconstruction SIR
        t    = np.arange(len(df))
        vals = df[i_col].values.astype(float)

        # Si valeurs croissantes --> cas cumulatifs --> differentier
        if vals[-1] > vals[0] * 1.5:
            new_cases = np.maximum(0, np.diff(vals, prepend=vals[0]))
        else:
            new_cases = np.maximum(0, vals)

        I = (pd.Series(new_cases)
             .rolling(window=self.D_I, min_periods=1)
             .sum().values)
        S = np.maximum(0, self.N - np.cumsum(new_cases))
        R = np.maximum(0, self.N - S - I)

        print(f"  -> {len(t)} points lus")
        print(f"  -> Pic I(t) : {I.max():,.0f}")

        return t, S, I, R, new_cases

    # -------------------------------------------------------------------------
    # Application du lissage
    # -------------------------------------------------------------------------
    def _smooth(self, arr):
        """Applique le filtre Savitzky-Golay et force les valeurs positives."""
        smoothed = savgol_filter(arr, SG_WINDOW, SG_POLYORDER)
        return np.clip(smoothed, 0, None)

    # -------------------------------------------------------------------------
    # METHODE PRINCIPALE : run()
    # -------------------------------------------------------------------------
    def run(self, plot=True, save_path="outputs/M1_cameroun.png"):
        """
        Charge, detecte, lit et lisse les donnees epidemiologiques.

        Fonctionne avec n'importe quel CSV en entree.

        Returns
        -------
        df_sir : pd.DataFrame -- serie temporelle SIR propre
        meta   : dict         -- metadonnees pour M2 et M3
        """
        print(f"\n[M1 Universel] Chargement : {self.data_path}")

        # Chargement brut
        df = pd.read_csv(self.data_path)
        print(f"  -> {len(df):,} lignes, {len(df.columns)} colonnes")
        print(f"  -> Colonnes : {list(df.columns)}")

        # Detection ou utilisation du mode fourni
        mode = self.mode
        if mode == "auto":
            mode = self._detect_format(df)

        # Lecture selon le mode detecte
        if mode == "gama":
            t, S, I, R, new_cases = self._read_gama_format(df)
        elif mode == "jhu":
            t, S, I, R, new_cases = self._read_jhu_format(df, self.state)
        else:
            t, S, I, R, new_cases = self._read_generic_format(df)

        # Lissage Savitzky-Golay
        S_smooth  = self._smooth(S)
        I_smooth  = self._smooth(I)
        R_smooth  = self._smooth(R)
        nc_smooth = self._smooth(new_cases)

        # Construction du DataFrame final
        self.df_sir = pd.DataFrame({
            't'          : t,
            'S_data'      : S_smooth,
            'I_data'      : I_smooth,
            'R_data'      : R_smooth,
            'S_raw'      : S,
            'I_raw'      : I,
            'R_raw'      : R,
            'new_cases'  : new_cases,
            'new_smooth' : nc_smooth,
        })

        # Metadonnees pour M2 et M3
        meta = {
            'state'      : self.state,
            'N'          : self.N,
            'D_I_prior'  : self.D_I,
            'T'          : len(self.df_sir),
            'peak_I'     : I_smooth.max(),
            'disease'    : 'COVID-19',
            't_obs'      : t.astype(float),
            'S_data'      : S_smooth,
            'I_data'      : I_smooth,
            'R_data'      : R_smooth,
        }

        if plot:
            self._plot(save_path)

        print("\n" + "=" * 65)
        print("  MODULE M1 UNIVERSEL -- TERMINE")
        print("=" * 65)
        print(f"  -> {meta['T']} points | Pic I = {meta['peak_I']:,.0f}")
        print(f"  -> Pret pour M2 et M3")
        print("=" * 65)

        return self.df_sir, meta

    # -------------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------------
    def _plot(self, save_path):
        """Graphique de controle qualite."""
        os.makedirs("outputs", exist_ok=True)
        df = self.df_sir

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"M1 Universel -- {self.state} | N={self.N:,}",
            fontsize=13, fontweight='bold'
        )

        # Nouveaux cas
        ax1 = axes[0]
        ax1.bar(df['t'], df['new_cases'],
                color='lightcoral', alpha=0.5, label='Brut')
        ax1.plot(df['t'], df['new_smooth'],
                 color='darkred', linewidth=2,
                 label='Lisse (SG)')
        ax1.set_title('Nouveaux cas journaliers')
        ax1.set_xlabel('Tick / Jour')
        ax1.set_ylabel('Nouveaux cas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Compartiments SIR
        ax2 = axes[1]
        ax2.plot(df['t'], df['S_data'] / self.N * 100,
                 color='green', linewidth=2, label='S(t) %N')
        ax2.plot(df['t'], df['I_data'] / self.N * 100,
                 color='red',   linewidth=2, label='I(t) %N')
        ax2.plot(df['t'], df['R_data'] / self.N * 100,
                 color='blue',  linewidth=2, label='R(t) %N')
        ax2.set_title('Compartiments SIR reconstruits')
        ax2.set_xlabel('Tick / Jour')
        ax2.set_ylabel('Fraction population (%N)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> Graphique : {save_path}")


# -----------------------------------------------------------------------------
# POINT D'ENTREE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST -- M1 Universel")
    print("=" * 65 + "\n")

    # Test avec le CSV JHU existant
    m1 = AcquisitionModuleUniversel(
        data_path = "data/usa_county_wise.csv",
        N         = 19_450_000,
        mode      = "auto",
        state     = "New York"
    )
    df_sir, meta = m1.run(plot=True)
    print(f"\nTest OK -- {meta['T']} points")

    # -----------------------------------------------------------------------
    # Pour utiliser avec le CSV de sortie GAMA camerounais :
    # -----------------------------------------------------------------------
    # m1 = AcquisitionModuleUniversel(
    #     data_path = "data/sir_output_cameroun.csv",
    #     N         = 5000,          # population de ta simulation GAMA
    #     mode      = "gama",        # format GAMA
    #     state     = "Cameroun"
    # )
    # df_sir, meta = m1.run(plot=True) // ceci c'est pour prochainement 