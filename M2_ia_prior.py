"""
=============================================================================
  MODULE M2 -- EXTRACTION DES PRIOR PAR IA GENERATIVE
=============================================================================

  Pipeline d'estimation automatique des parametres SIR (alpha, beta', gamma)
  par Intelligence Artificielle generative

  Auteur  : Henri (Master 2 - GI)
  IA      : Google Gemini (gemini-1.5-flash) via API gratuite
  Maladie : COVID-19 (SARS-CoV-2)

  Role de ce module :
    - Interroger l'IA generative (Gemini) pour extraire les valeurs
      epidemiologiques a priori (prior) de la maladie modelisee
    - Convertir ces prior en bornes d'initialisation pour l'optimiseur
    - Detecter automatiquement les vagues epidemiques dans la serie
    - Fournir un point de depart biologiquement plausible au Module M3

  Concept cle -- Le "prior" :
    Un prior est la connaissance qu'on a sur une maladie AVANT de
    regarder les donnees. Par exemple, on sait que le COVID-19 a un
    R0 entre 2 et 3, une periode infectieuse de 5 a 14 jours.
    L'IA extrait ces informations de la litterature scientifique
    pour initialiser intelligemment notre optimiseur PSO.

    Sans prior : optimiseur cherche dans [0.001, 0.99] -- lent, instable
    Avec prior  : optimiseur commence pres de la vraie solution -- rapide

  Pourquoi Gemini ?
    - API gratuite (15 requetes/min, 1500/jour) -- suffisant pour le projet
    - Modele gemini-1.5-flash : rapide et precis pour les taches JSON
    - Aucune carte bancaire requise pour commencer

=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import os                              # Acces aux variables d'environnement
import json                            # Parsing des reponses JSON de l'IA
import numpy as np                     # Calculs numeriques
import matplotlib.pyplot as plt        # Visualisation
import matplotlib.dates as mdates      # Formatage des axes temporels
import pandas as pd                    # Manipulation des donnees
from google import genai               # Google Gemini (google-genai)
import anthropic                           # Anthropic Claude (optionnel)
from dotenv import load_dotenv         # Chargement du fichier .env
import warnings
warnings.filterwarnings('ignore')

# Chargement des variables d'environnement depuis le fichier .env 
# Cela charge GEMINI_API_KEY sans la mettre en dur dans le code
load_dotenv()


# -----------------------------------------------------------------------------
# CONSTANTES GLOBALES
# -----------------------------------------------------------------------------

# Modele Gemini utilise
# gemini-1.5-flash : modele gratuit, rapide, tres bon pour les taches JSON
# gemini-1.5-pro   : modele plus puissant mais quota plus limite
# Modeles disponibles -- le pipeline essaie dans cet ordre :
# 1. Anthropic Claude (si cle valide et credits disponibles)
# 2. Google Gemini   (si cle valide et quota disponible)
# 3. Fallback manuel (valeurs litterature OMS/CDC -- toujours disponible)
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
GEMINI_MODEL    = "gemini-2.0-flash"

# Prior par defaut si l'API est indisponible (fallback de securite)
# Ces valeurs sont issues de la litterature COVID-19 (OMS, CDC 2020)
# Elles servent de filet de securite si Gemini ne repond pas
DEFAULT_PRIOR_COVID = {
    "R0"           : {"mean": 2.5,  "ci95_low": 2.0,  "ci95_high": 3.5},
    "D_I"          : {"mean": 10.0, "ci95_low": 5.0,  "ci95_high": 14.0},
    "alpha_0"      : {"mean": 0.20, "ci95_low": 0.10, "ci95_high": 0.35},
    "beta_prime_0" : {"mean": 7,    "ci95_low": 5,    "ci95_high": 14},
    "gamma_0"      : {"mean": 0.10, "ci95_low": 0.07, "ci95_high": 0.20},
    "source"       : "Fallback -- Litterature COVID-19 OMS/CDC 2020",
    "confidence"   : "medium"
}

# Prior manuel pour la maladie a virus Ebola (epidemie Afrique de l'Ouest
# 2014-2015 : Guinee, Liberia, Sierra Leone). Valeurs issues de la
# litterature epidemiologique specifique a Ebola (tres differentes du
# COVID-19 : R0 plus faible, periode infectieuse beaucoup plus courte) :
#   - R0 par pays (Althaus, C. L. (2014). Estimating the reproduction
#     number of Ebola virus (EBOV) during the 2014 outbreak in West
#     Africa. PLOS Currents Outbreaks) : Guinee 1.51 [1.50-1.52],
#     Liberia 1.59 [1.57-1.60], Sierra Leone 2.53 [2.41-2.67] -- on
#     retient ici une moyenne agregee representative des 3 pays.
#   - Periode infectieuse 1/gamma ~ 5.61 j (Legrand et al. 2007, reprise
#     des estimations Ebola Zaire 1995) et fourchette 3-10 j (WHO Ebola
#     Response Team / Van Kerkhove et al. (2015). A review of
#     epidemiological parameters from Ebola outbreaks to inform early
#     public health decision-making. Scientific Data, 2, 150019).
DEFAULT_PRIOR_EBOLA = {
    "R0"           : {"mean": 1.8,  "ci95_low": 1.5,  "ci95_high": 2.5},
    "D_I"          : {"mean": 7.0,  "ci95_low": 3.0,  "ci95_high": 10.0},
    "alpha_0"      : {"mean": 0.25, "ci95_low": 0.15, "ci95_high": 0.40},
    "beta_prime_0" : {"mean": 6,    "ci95_low": 3,    "ci95_high": 10},
    "gamma_0"      : {"mean": 0.14, "ci95_low": 0.10, "ci95_high": 0.30},
    "source"       : ("Fallback -- Litterature Ebola Afrique de l'Ouest 2014-2015 "
                       "(Althaus 2014 ; WHO Ebola Response Team, NEJM 2014 ; "
                       "Legrand et al. 2007 ; Van Kerkhove et al. 2015)"),
    "confidence"   : "medium"
}

# Registre des prior manuels par maladie -- selection par sous-chaine
# insensible a la casse dans le nom de la maladie transmis au module.
# Toute maladie non repertoriee retombe sur le prior COVID-19 (comportement
# historique du pipeline, conserve pour ne rien casser).
DEFAULT_PRIOR_REGISTRY = {
    "ebola" : DEFAULT_PRIOR_EBOLA,
    "covid" : DEFAULT_PRIOR_COVID,
}


def select_default_prior(disease):
    """Retourne le prior de repli (fallback) adapte a la maladie donnee."""
    disease_lower = (disease or "").lower()
    for key, prior in DEFAULT_PRIOR_REGISTRY.items():
        if key in disease_lower:
            return prior
    return DEFAULT_PRIOR_COVID


# -----------------------------------------------------------------------------
# CLASSE PRINCIPALE : IAPriorModule
# -----------------------------------------------------------------------------
class IAPriorModule:
    """
    Module M2 du pipeline de calibration SIR.

    Ce module utilise l'IA generative (Google Gemini) pour extraire
    automatiquement les valeurs a priori (prior) des parametres
    epidemiologiques d'une maladie donnee, a partir de la litterature
    scientifique encodee dans le LLM.

    Le prior sert a initialiser intelligemment l'optimiseur PSO du
    Module M3, en reduisant l'espace de recherche aux valeurs
    biologiquement plausibles.

    Attributs
    ---------
    disease   : str   -- nom de la maladie a analyser
    state     : str   -- zone geographique
    N         : int   -- taille de la population
    meta      : dict  -- metadonnees issues du Module M1
    model     : genai.GenerativeModel -- modele Gemini configure
    prior     : dict  -- valeurs a priori extraites par l'IA
    waves     : list  -- vagues epidemiques detectees
    theta_0   : dict  -- point de depart de l'optimiseur
    bounds    : list  -- bornes admissibles pour l'optimiseur
    """

    def __init__(self, meta, disease="COVID-19"):
        """
        Initialise le module M2 avec les metadonnees du Module M1.

        Parameters
        ----------
        meta    : dict -- metadonnees issues de M1.run()
        disease : str  -- nom de la maladie (pour le prompt IA)
        """
        self.disease = disease
        self.state   = meta.get('state', 'Unknown')
        self.N       = meta.get('N', 1_000_000)
        self.meta    = meta
        self.prior   = None
        self.waves   = []
        self.theta_0 = None
        self.bounds  = None

        # Configuration des clients IA -- architecture hybride
        # Priorite 1 : Anthropic Claude (meilleure qualite)
        # Priorite 2 : Google Gemini   (gratuit)
        # Priorite 3 : Fallback manuel (toujours disponible)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        gemini_key    = os.environ.get("GEMINI_API_KEY")

        # Client Anthropic (optionnel -- necessite des credits)
        self.anthropic_client = None
        if anthropic_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print(f"  -> Anthropic Claude  : disponible (cle trouvee)")
            except Exception:
                print(f"  -> Anthropic Claude  : indisponible")

        # Client Gemini (optionnel -- gratuit avec quota)
        self.gemini_client = None
        if gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_key)
                print(f"  -> Google Gemini     : disponible (cle trouvee)")
            except Exception:
                print(f"  -> Google Gemini     : indisponible")

        if not anthropic_key and not gemini_key:
            print(f"  -> Aucune cle API -- fallback manuel (OMS/CDC 2020)")

        print("=" * 65)
        print("  MODULE M2 -- Extraction des Prior par IA Generative")
        print("=" * 65)
        print(f"  Maladie       : {self.disease}")
        print(f"  Zone          : {self.state}")
        print(f"  Population N  : {self.N:,}")
        print(f"  Strategie IA  : Anthropic -> Gemini -> Fallback OMS")
        print("=" * 65)

    # -------------------------------------------------------------------------
    # METHODE INTERNE : appel securise a l'API Gemini
    # -------------------------------------------------------------------------
    def _call_gemini(self, prompt, label=""):
        """
        Effectue un appel IA hybride avec cascade de fallback :

        Niveau 1 -- Anthropic Claude (meilleure qualite scientifique)
          Utilise si ANTHROPIC_API_KEY est definie et credits disponibles.
          Modele : claude-sonnet-4-20250514

        Niveau 2 -- Google Gemini (gratuit, quota journalier)
          Utilise si GEMINI_API_KEY est definie et quota disponible.
          Modele : gemini-2.0-flash

        Niveau 3 -- Fallback manuel
          Valeurs issues de la litterature OMS/CDC 2020.
          Toujours disponible, garantit que le pipeline tourne.

        Parameters
        ----------
        prompt : str -- le prompt a envoyer a l'IA
        label  : str -- libelle pour le logging

        Returns
        -------
        text : str ou None -- texte de la reponse, None si echec total
        """

        # --- Niveau 1 : Anthropic Claude ---
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model      = ANTHROPIC_MODEL,
                    max_tokens = 1024,
                    system     = ("Tu es un epidemiologiste expert. "
                                  "Reponds UNIQUEMENT en JSON valide, "
                                  "sans texte ni markdown."),
                    messages   = [{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
                print(f"  -> [Anthropic] Reponse recue ({len(text)} car.)")
                return text
            except Exception as e:
                print(f"  !! [Anthropic] Erreur : {type(e).__name__} -- "
                      f"bascule sur Gemini")

        # --- Niveau 2 : Google Gemini ---
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model    = GEMINI_MODEL,
                    contents = prompt,
                    config   = genai.types.GenerateContentConfig(
                        temperature       = 0.1,
                        max_output_tokens = 1024,
                    )
                )
                text = response.text.strip()
                print(f"  -> [Gemini] Reponse recue ({len(text)} car.)")
                return text
            except Exception as e:
                print(f"  !! [Gemini] Erreur : {type(e).__name__} -- "
                      f"bascule sur fallback")

        # --- Niveau 3 : Fallback ---
        print(f"  -> [{label}] Aucune IA disponible -- fallback manuel")
        return None

    # -------------------------------------------------------------------------
    # METHODE INTERNE : nettoyage et parsing JSON
    # -------------------------------------------------------------------------
    def _parse_json(self, raw_text, fallback, label=""):
        """
        Nettoie et parse la reponse JSON de l'IA.

        Les LLM ajoutent parfois des balises markdown autour du JSON
        (comme ```json ... ```). Cette methode les supprime avant
        le parsing pour garantir un parsing robuste.

        Parameters
        ----------
        raw_text : str  -- texte brut retourne par l'IA
        fallback : dict -- dictionnaire a retourner en cas d'echec
        label    : str  -- libelle pour le logging

        Returns
        -------
        data : dict -- donnees parsees ou fallback
        """
        if raw_text is None:
            print(f"  -> [{label}] Utilisation du fallback (reponse vide)")
            return fallback

        # Nettoyage des balises markdown eventuelles
        clean = raw_text
        for tag in ['```json', '```JSON', '```']:
            clean = clean.replace(tag, '')
        clean = clean.strip()

        try:
            data = json.loads(clean)
            print(f"  -> [{label}] JSON parse avec succes")
            return data
        except json.JSONDecodeError as e:
            print(f"  !! [{label}] Erreur parsing JSON : {e}")
            print(f"  -> [{label}] Utilisation du fallback")
            return fallback

    # -------------------------------------------------------------------------
    # ETAPE 1 : Extraction des prior epidemiologiques par l'IA
    # -------------------------------------------------------------------------
    def extract_prior_from_llm(self):
        """
        Interroge Gemini pour extraire les parametres epidemiologiques
        a priori de la maladie depuis la litterature scientifique.

        Technique utilisee : Zero-Shot Chain-of-Thought Prompting
        - On demande a l'IA de raisonner avant de repondre
        - La reponse est forcee en JSON structure pour un parsing fiable
        - temperature=0.1 garantit des reponses reproductibles

        Le prompt est concu pour extraire exactement les valeurs dont
        l'optimiseur PSO a besoin pour demarrer intelligemment.

        Pourquoi l'IA ici ?
        Gemini a ete entraine sur des milliers d'articles scientifiques.
        Il peut donc nous donner des valeurs de R0, duree infectieuse,
        etc. directement depuis la litterature, sans recherche manuelle.
        C'est la valeur ajoutee centrale de l'IA dans notre pipeline.

        Returns
        -------
        self -- chaine les methodes (pattern fluent interface)
        """
        print(f"\n[M2 - Etape 1] Interrogation de Gemini ({GEMINI_MODEL})...")
        print(f"  -> Extraction des prior pour : {self.disease}")

        # Construction du prompt
        # On force le JSON pur sans markdown pour faciliter le parsing
        prompt = f"""Tu es un epidemiologiste expert.
Pour la maladie {self.disease} dans la region {self.state},
fournis les parametres epidemiologiques avec intervalles de confiance 95%.

Ces parametres serviront a calibrer un modele SIR ou :
  - alpha = infection_rate = probabilite de transmission par contact (0 a 1)
  - beta_prime = duree minimale periode infectieuse en jours (entier)
  - gamma = recovery_rate = taux de guerison par jour (0 a 1)
  - R0 = alpha / gamma = nombre de reproduction de base

Reponds UNIQUEMENT avec du JSON valide, sans texte avant ou apres,
sans balises markdown. Juste le JSON :
{{
  "disease": "{self.disease}",
  "region": "{self.state}",
  "R0": {{
    "mean": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "interpretation": "texte"
  }},
  "D_I": {{
    "mean": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0,
    "unit": "jours",
    "interpretation": "texte"
  }},
  "alpha_0": {{
    "mean": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0
  }},
  "beta_prime_0": {{
    "mean": 0,
    "ci95_low": 0,
    "ci95_high": 0,
    "unit": "jours"
  }},
  "gamma_0": {{
    "mean": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.0
  }},
  "period": "2020",
  "variant": "souche originale",
  "source": "references",
  "confidence": "high",
  "notes": "observations"
}}"""

        # Appel a l'API Gemini
        raw_text   = self._call_gemini(prompt, label="prior")
        self.prior = self._parse_json(raw_text, select_default_prior(self.disease), "prior")

        # Affichage des valeurs extraites
        print(f"  -> Confiance IA  : {self.prior.get('confidence', 'N/A')}")
        print(f"  -> R0 estime     : "
              f"{self.prior['R0']['mean']} "
              f"[{self.prior['R0']['ci95_low']}, "
              f"{self.prior['R0']['ci95_high']}]")
        print(f"  -> D_I estime    : {self.prior['D_I']['mean']} jours")
        print(f"  -> alpha_0       : {self.prior['alpha_0']['mean']}")
        print(f"  -> beta_prime_0  : {self.prior['beta_prime_0']['mean']} jours")
        print(f"  -> gamma_0       : {self.prior['gamma_0']['mean']}")

        return self

    # -------------------------------------------------------------------------
    # ETAPE 2 : Detection automatique des vagues epidemiques
    # -------------------------------------------------------------------------
    def detect_epidemic_waves(self):
        """
        Interroge Gemini pour detecter les vagues epidemiques successives
        dans la serie temporelle des nouveaux cas.

        Pourquoi detecter les vagues ?
        Une epidemie peut avoir plusieurs vagues (ex: COVID-19 vague 1,
        vague 2, vague delta...). Chaque vague a des parametres differents
        (R0, alpha, gamma) dus aux variants, mesures sanitaires, immunite.
        Calibrer toute la serie comme une seule vague produit de mauvais
        resultats. La detection par l'IA permet de segmenter la serie.

        On envoie a Gemini un resume compact de la serie (20 points)
        pour economiser les tokens et rester dans le quota gratuit.

        Returns
        -------
        self -- chaine les methodes
        """
        print(f"\n[M2 - Etape 2] Detection des vagues epidemiques par IA...")

        I_data = self.meta.get('I_data', np.array([]))
        t_obs = self.meta.get('t_obs', np.array([]))
        T     = len(I_data)

        if T == 0:
            print("  -> Aucune serie -- vague unique supposee")
            self.waves = [{"wave": 1, "start": 0, "peak": 0, "end": 0,
                           "R0_approx": self.prior['R0']['mean'],
                           "description": "Vague unique"}]
            return self

        # Resume compact de la serie (20 points max)
        # On ne peut pas envoyer 188 valeurs a l'API -- trop de tokens
        n_summary = min(20, T)
        indices   = np.linspace(0, T-1, n_summary, dtype=int)
        summary   = [(int(t_obs[i]), int(round(I_data[i])))
                     for i in indices]

        peak_day = int(np.argmax(I_data))
        peak_val = int(round(np.max(I_data)))

        prompt = f"""Tu es un epidemiologiste expert.
Analyse cette serie temporelle de {self.disease} dans {self.state}.
Donnees (jour, infectes_actifs) sur {T} jours : {summary}
Pic principal : jour {peak_day} avec {peak_val:,} infectes actifs.

Identifie les vagues epidemiques successives.
Reponds UNIQUEMENT avec du JSON valide, sans texte ni markdown :
{{
  "n_waves": 1,
  "waves": [
    {{
      "wave": 1,
      "start": 0,
      "peak": {peak_day},
      "end": {T-1},
      "R0_approx": 2.5,
      "description": "Premiere vague COVID-19"
    }}
  ],
  "analysis": "interpretation globale en une phrase"
}}"""

        # Fallback vague unique
        fallback_waves = {
            "n_waves" : 1,
            "waves"   : [{
                "wave"        : 1,
                "start"       : 0,
                "peak"        : peak_day,
                "end"         : T - 1,
                "R0_approx"   : self.prior['R0']['mean'],
                "description" : "Vague unique -- fallback"
            }],
            "analysis": "Une seule vague detectee par defaut"
        }

        raw_text  = self._call_gemini(prompt, label="vagues")
        wave_data = self._parse_json(raw_text, fallback_waves, "vagues")

        self.waves = wave_data.get('waves', fallback_waves['waves'])
        n_waves    = wave_data.get('n_waves', 1)
        analysis   = wave_data.get('analysis', '')

        print(f"  -> {n_waves} vague(s) detectee(s)")
        for w in self.waves:
            print(f"     Vague {w['wave']} : "
                  f"jours {w['start']} -> {w['peak']} (pic) -> {w['end']} "
                  f"| R0 approx {w['R0_approx']}")
        if analysis:
            print(f"  -> Analyse : {str(analysis)[:100]}...")

        return self

    # -------------------------------------------------------------------------
    # ETAPE 3 : Calcul du point de depart theta_0 et des bornes pour M3
    # -------------------------------------------------------------------------
    def compute_optimizer_initialization(self):
        """
        Convertit les prior extraits par l'IA en parametres concrets
        pour initialiser l'optimiseur PSO du Module M3.

        Formules de conversion prior -> parametres GAMA :

          gamma_0 = 1 / D_I_mean
            Taux de guerison = inverse de la duree infectieuse.
            Exemple : D_I = 10 jours => gamma_0 = 0.10 / jour

          alpha_0 = R0_mean * gamma_0
            Taux d'infection = R0 x taux de guerison.
            Relation fondamentale : R0 = alpha / gamma
            Exemple : R0=2.5, gamma=0.10 => alpha=0.25

          beta_prime_0 = D_I_mean (arrondi a l'entier)
            Duree minimale infectieuse en ticks (jours GAMA).
            Exemple : D_I = 10 jours => beta_prime_0 = 10

        Les bornes Omega sont elargies de 20% au-dela des IC95%
        pour laisser a l'optimiseur une marge d'exploration.

        Returns
        -------
        self -- chaine les methodes
        """
        print(f"\n[M2 - Etape 3] Calcul du point de depart theta_0...")

        p = self.prior

        # Calcul de gamma_0 : taux de guerison journalier
        gamma_0      = 1.0 / p['D_I']['mean']
        gamma_0_low  = 1.0 / p['D_I']['ci95_high']
        gamma_0_high = 1.0 / p['D_I']['ci95_low']

        # Calcul de alpha_0 : taux de transmission
        # Relation fondamentale SIR : R0 = alpha / gamma
        # Donc : alpha = R0 * gamma
        alpha_0      = p['R0']['mean']      * gamma_0
        alpha_0_low  = p['R0']['ci95_low']  * gamma_0_low
        alpha_0_high = p['R0']['ci95_high'] * gamma_0_high

        # Calcul de beta_prime_0 : duree minimale infectieuse (entier)
        beta_prime_0 = int(round(p['D_I']['mean']))
        beta_p_low   = max(1, int(p['D_I']['ci95_low']))
        beta_p_high  = int(p['D_I']['ci95_high']) + 2

        # Stockage du point de depart theta_0
        self.theta_0 = {
            'alpha_0'      : round(alpha_0,   4),
            'beta_prime_0' : beta_prime_0,
            'gamma_0'      : round(gamma_0,   4),
            'R0_implied'   : round(alpha_0 / gamma_0, 3)
        }

        # Definition des bornes de l'espace de recherche Omega
        # Marge de 20% au-dela des IC95% pour l'exploration
        MARGIN   = 0.20
        alpha_lb = max(0.001, alpha_0_low  * (1 - MARGIN))
        alpha_ub = min(0.990, alpha_0_high * (1 + MARGIN))
        beta_lb  = max(1,     beta_p_low  - 1)
        beta_ub  = min(30,    beta_p_high + 2)
        gamma_lb = max(0.001, gamma_0_low  * (1 - MARGIN))
        gamma_ub = min(0.500, gamma_0_high * (1 + MARGIN))

        # Format scipy.optimize : liste de tuples (min, max)
        self.bounds = [
            (alpha_lb, alpha_ub),
            (beta_lb,  beta_ub),
            (gamma_lb, gamma_ub)
        ]

        # Affichage du tableau recapitulatif
        print(f"\n  POINT DE DEPART theta_0 (issu du prior IA) :")
        print(f"  {'Parametre':<20} {'Valeur':<12} {'Borne min':<12} {'Borne max'}")
        print(f"  {'-'*57}")
        print(f"  {'alpha_0':<20} {self.theta_0['alpha_0']:<12.4f} "
              f"{alpha_lb:<12.4f} {alpha_ub:.4f}")
        print(f"  {'beta_prime_0 (jours)':<20} {self.theta_0['beta_prime_0']:<12} "
              f"{beta_lb:<12} {beta_ub}")
        print(f"  {'gamma_0':<20} {self.theta_0['gamma_0']:<12.4f} "
              f"{gamma_lb:<12.4f} {gamma_ub:.4f}")
        print(f"  {'-'*57}")
        print(f"  R0 implique : {self.theta_0['R0_implied']} "
              f"(prior IA : {p['R0']['mean']})")

        return self

    # -------------------------------------------------------------------------
    # ETAPE 4 : Rapport narratif scientifique
    # -------------------------------------------------------------------------
    def generate_narrative_report(self):
        """
        Demande a Gemini de generer un rapport narratif d'interpretation
        epidemiologique des donnees et des prior extraits.

        Ce rapport est directement utilisable dans le memoire de Master.
        Il illustre concrètement l'apport de l'IA generative au pipeline :
        non seulement extraction de donnees, mais aussi redaction
        scientifique automatisee.

        Returns
        -------
        report : str -- rapport narratif en francais (150-200 mots)
        """
        print(f"\n[M2 - Etape 4] Generation du rapport narratif IA...")

        p     = self.prior
        theta = self.theta_0

        prompt = f"""Tu es un epidemiologiste expert. Redige un paragraphe
scientifique concis (150-200 mots) en francais sur cette situation :

Maladie    : {self.disease}
Region     : {self.state} (N = {self.N:,} habitants)
Periode    : Janvier - Juillet 2020
Parametres estimes par prior IA :
  - R0 = {p['R0']['mean']} (IC95% : [{p['R0']['ci95_low']}, {p['R0']['ci95_high']}])
  - Duree infectieuse D_I = {p['D_I']['mean']} jours
  - alpha_0 = {theta['alpha_0']} (taux de transmission)
  - gamma_0 = {theta['gamma_0']} (taux de guerison)
  - Vagues detectees : {len(self.waves)}

Mentionne le R0, l'interpretation epidemiologique et le contexte
de la premiere vague COVID-19 aux Etats-Unis en 2020.
Style : scientifique academique, niveau memoire de Master 2."""

        raw_text = self._call_gemini(prompt, label="rapport")

        if raw_text:
            print(f"  -> Rapport genere ({len(raw_text)} caracteres)")
            print(f"\n  RAPPORT NARRATIF IA :")
            print(f"  {'-'*60}")
            for line in raw_text.split('\n'):
                print(f"  {line}")
            print(f"  {'-'*60}")
            return raw_text
        else:
            msg = "Rapport non disponible (erreur API Gemini)"
            print(f"  -> {msg}")
            return msg

    # -------------------------------------------------------------------------
    # VISUALISATION
    # -------------------------------------------------------------------------
    def plot_prior_and_waves(self, df_sir,
                              save_path="outputs/M2_prior_waves.png"):
        """
        Genere un graphique en deux panneaux :

        Panneau gauche : Serie temporelle I(t) avec les vagues detectees
          - Courbe des infectes actifs lissee
          - Bandes colorees pour chaque vague
          - Lignes verticales aux pics de chaque vague

        Panneau droit : Intervalles de confiance des prior extraits
          - Barres pour alpha_0, gamma_0, R0
          - Barres d'erreur representant les IC 95%
          - Valeurs numeriques annotees sur les barres

        Parameters
        ----------
        df_sir    : pd.DataFrame -- donnees SIR du Module M1
        save_path : str          -- chemin de sauvegarde du graphique
        """
        os.makedirs("outputs", exist_ok=True)

        p     = self.prior
        theta = self.theta_0

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"Module M2 -- Prior IA (Gemini) et Vagues epidemiques\n"
            f"{self.disease} | {self.state} | Modele : {GEMINI_MODEL}",
            fontsize=13, fontweight='bold'
        )

        # -- Panneau gauche : I(t) avec vagues ---------------------
        ax1   = axes[0]
        dates = df_sir['Date'] if 'Date' in df_sir.columns else pd.Series(df_sir['t'].values) if 'Date' in df_sir.columns else df_sir['t']

        ax1.fill_between(dates, df_sir['I_data'],
                         alpha=0.3, color='red',
                         label='I(t) infectes actifs')
        ax1.plot(dates, df_sir['I_data'],
                 color='darkred', linewidth=2)

        # Annotation de chaque vague detectee
        colors_wave = ['blue', 'green', 'orange', 'purple']
        for w in self.waves:
            c = colors_wave[(w['wave'] - 1) % len(colors_wave)]
            if w['start'] < len(dates) and w['end'] < len(dates):
                # Zone coloree representant la duree de la vague
                ax1.axvspan(
                    dates.iloc[w['start']],
                    dates.iloc[min(w['end'], len(dates)-1)],
                    alpha=0.12, color=c,
                    label=f"Vague {w['wave']} (R0~{w['R0_approx']})"
                )
                # Ligne verticale au pic de la vague
                if w['peak'] < len(dates):
                    ax1.axvline(
                        dates.iloc[w['peak']],
                        color=c, linestyle='--',
                        linewidth=1.5, alpha=0.8
                    )
                    ax1.annotate(
                        f"Pic V{w['wave']}\nJ{w['peak']}",
                        xy=(dates.iloc[w['peak']],
                            df_sir['I_data'].iloc[w['peak']]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9, color=c, fontweight='bold'
                    )

        ax1.set_title('Infectes actifs I(t) -- Vagues detectees par IA',
                      fontsize=11)
        ax1.set_ylabel('Infectes actifs')
        ax1.set_xlabel('Date')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.grid(True, alpha=0.3)

        # -- Panneau droit : intervalles de confiance des prior ----
        ax2    = axes[1]
        params = ['alpha_0\n(infection)', 'gamma_0\n(guerison)', 'R0']
        means  = [theta['alpha_0'], theta['gamma_0'], p['R0']['mean']]
        lows   = [self.bounds[0][0], self.bounds[2][0],
                  p['R0']['ci95_low']]
        highs  = [self.bounds[0][1], self.bounds[2][1],
                  p['R0']['ci95_high']]
        colors = ['#E74C3C', '#2ECC71', '#3498DB']

        x    = np.arange(len(params))
        bars = ax2.bar(x, means, color=colors, alpha=0.75, width=0.5,
                       label='Valeur centrale (prior IA)')

        # Barres d'erreur = intervalles de confiance a 95%
        ax2.errorbar(
            x, means,
            yerr=[
                [m - l for m, l in zip(means, lows)],
                [h - m for m, h in zip(means, highs)]
            ],
            fmt='none', color='black',
            capsize=8, capthick=2, linewidth=2,
            label='IC 95% (prior IA)'
        )

        # Annotation des valeurs numeriques sur les barres
        for bar, m in zip(bars, means):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() * 1.05,
                f'{m:.3f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        ax2.set_title(
            'Prior epidemiologiques extraits par IA Gemini\n'
            '(valeurs centrales + IC 95%)',
            fontsize=11
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(params, fontsize=11)
        ax2.set_ylabel('Valeur du parametre')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # Source des prior
        source_text = str(p.get('source', 'IA generative Gemini'))[:70]
        ax2.text(
            0.5, -0.18, f"Source : {source_text}",
            transform=ax2.transAxes,
            ha='center', fontsize=8, color='gray', style='italic'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  -> Graphique sauvegarde : {save_path}")

    # -------------------------------------------------------------------------
    # METHODE PRINCIPALE : run()
    # -------------------------------------------------------------------------
    def run(self, df_sir, plot=True, save_path="outputs/M2_prior_waves.png"):
        """
        Orchestre toutes les etapes du Module M2 en sequence.

        Ordre d'execution :
        1. Extraction des prior epidemiologiques par Gemini
        2. Detection automatique des vagues epidemiques
        3. Calcul du point de depart theta_0 et bornes pour M3
        4. Generation du rapport narratif scientifique
        5. Visualisation (optionnel)

        Parameters
        ----------
        df_sir : pd.DataFrame -- donnees SIR issues du Module M1
        plot   : bool         -- si True, genere les graphiques

        Returns
        -------
        prior_result : dict -- tout ce dont M3 a besoin pour demarrer
        """
        # Execution sequentielle des 4 etapes
        self.extract_prior_from_llm()
        self.detect_epidemic_waves()
        self.compute_optimizer_initialization()
        report = self.generate_narrative_report()

        if plot:
            self.plot_prior_and_waves(df_sir, save_path=save_path)

        # Dictionnaire de sortie complet pour le Module M3
        prior_result = {
            # Point de depart de l'optimiseur PSO
            'theta_0'      : self.theta_0,

            # Bornes de l'espace de recherche Omega
            # Format : [(alpha_min, alpha_max),
            #            (beta_min,  beta_max),
            #            (gamma_min, gamma_max)]
            'bounds'       : self.bounds,

            # Valeurs centrales (acces direct pour M3)
            'alpha_0'      : self.theta_0['alpha_0'],
            'beta_prime_0' : self.theta_0['beta_prime_0'],
            'gamma_0'      : self.theta_0['gamma_0'],

            # Informations epidemiologiques
            'R0_prior'     : self.prior['R0']['mean'],
            'R0_ci95'      : [self.prior['R0']['ci95_low'],
                               self.prior['R0']['ci95_high']],
            'D_I_prior'    : self.prior['D_I']['mean'],

            # Vagues detectees par l'IA
            'waves'        : self.waves,
            'n_waves'      : len(self.waves),

            # Rapport narratif genere par Gemini
            'narrative'    : report,

            # Metadonnees
            'disease'      : self.disease,
            'state'        : self.state,
            'model_used'   : (ANTHROPIC_MODEL if self.anthropic_client else GEMINI_MODEL if self.gemini_client else 'Fallback-OMS'),
            'confidence'   : self.prior.get('confidence', 'N/A'),
            'source'       : self.prior.get('source', 'N/A'),
        }

        print("\n" + "=" * 65)
        print("  MODULE M2 -- TERMINE AVEC SUCCES")
        print("=" * 65)
        print(f"  -> theta_0 = (alpha={prior_result['alpha_0']}, "
              f"beta'={prior_result['beta_prime_0']}, "
              f"gamma={prior_result['gamma_0']})")
        print(f"  -> R0 prior  = {prior_result['R0_prior']}")
        print(f"  -> {prior_result['n_waves']} vague(s) detectee(s)")
        print(f"  -> Pret pour Module M3 (calibration PSO + Nelder-Mead)")
        print("=" * 65)

        return prior_result


# -----------------------------------------------------------------------------
# POINT D'ENTREE : test autonome
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TEST AUTONOME -- Module M2 IA Prior (Gemini)")
    print("=" * 65 + "\n")

    import sys
    sys.path.insert(0, '.')
    from M1_acquisition import AcquisitionModule

    # Execution M1
    m1 = AcquisitionModule(
        data_path = "data/usa_county_wise.csv",
        state     = "New York",
        N         = 19_450_000,
        D_I       = 14
    )
    df_sir, meta = m1.run(plot=False)

    # Execution M2
    m2 = IAPriorModule(meta=meta, disease="COVID-19")
    prior_result = m2.run(df_sir=df_sir, plot=True)

    # Affichage du resume
    print("\n=== RESUME PRIOR POUR MODULE M3 ===")
    print(f"  alpha_0      : {prior_result['alpha_0']}")
    print(f"  beta_prime_0 : {prior_result['beta_prime_0']} jours")
    print(f"  gamma_0      : {prior_result['gamma_0']}")
    print(f"  R0 prior     : {prior_result['R0_prior']}")
    print(f"  Bornes alpha : {prior_result['bounds'][0]}")
    print(f"  Bornes beta' : {prior_result['bounds'][1]}")
    print(f"  Bornes gamma : {prior_result['bounds'][2]}")
    print(f"  Vagues       : {prior_result['n_waves']}")
    print(f"  Confiance IA : {prior_result['confidence']}")
    print(f"\n-> Module M2 valide -- pret pour M3")


