"""
╔══════════════════════════════════════════════════════════════╗
║         OOH CAMPAIGN MODELER — Multi-Agent System           ║
║         Versione completa con integrazione MySQL             ║
╠══════════════════════════════════════════════════════════════╣
║  Dipendenze:                                                 ║
║      pip install pandas numpy scipy mysql-connector-python   ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import warnings
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURAZIONE — modifica qui i tuoi parametri
# ================================================================

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "",
    "database": "dbooh",
}

CAMPAGNE_TEST  = [1, 2, 3, 4, 5]   # idCampagna da analizzare
GIORNI         = 14                  # durata flat campagna
TARGET_ETA     = "18_34"            # "18_34" | "35_54" | "55plus" | None
N_SIMULAZIONI  = 1000               # iterazioni Monte Carlo per IC 95%
OUTPUT_CSV     = "risultati_ooh.csv"
VERBOSO        = False               # True = log dettagliato agente per agente


# ================================================================
# FORMATO ITALIANO PER I NUMERI
# ================================================================

def fmt_num(valore, decimali: int = 0) -> str:
    """
    Formatta un numero in stile italiano:
      separatore migliaia → punto  (.)
      separatore decimali → virgola (,)
    Es: 1234567.89 → "1.234.567,89"
    """
    if valore is None or (isinstance(valore, float) and math.isnan(valore)):
        return ""
    if decimali == 0:
        return f"{int(round(valore)):,}".replace(",", ".")
    else:
        s = f"{{:,.{decimali}f}}".format(valore)
        # swap: prima le virgole migliaia in X, poi il punto decimale in virgola,
        # poi X in punto
        return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(valore, decimali: int = 1) -> str:
    """Formatta percentuale in stile italiano. Es: 12.34 → "12,3%" """
    if valore is None:
        return ""
    return fmt_num(valore, decimali) + "%"


def dataframe_to_csv_ita(df: pd.DataFrame, path: Path):
    """
    Salva un DataFrame in CSV formato italiano:
      separatore colonne : punto e virgola (;)
      separatore decimali: virgola (,)
      separatore migliaia: punto (.)
      encoding           : utf-8-sig  ← Excel italiano lo apre correttamente
    """
    df_out = df.copy()

    # Float → stringa italiana
    for col in df_out.select_dtypes(include=["float64", "float32"]).columns:
        df_out[col] = df_out[col].apply(
            lambda x: fmt_num(x, 2) if pd.notna(x) else ""
        )
    # Int → stringa italiana con separatore migliaia
    for col in df_out.select_dtypes(include=["int64", "int32"]).columns:
        df_out[col] = df_out[col].apply(lambda x: fmt_num(x, 0))

    df_out.to_csv(
        path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"  Export → {path}")
    print(f"  Righe: {len(df_out)}  |  Colonne: {len(df_out.columns)}")
    print(f"  Formato: sep=;  decimali=,  migliaia=.  encoding=utf-8-sig")


# ================================================================
# STRUTTURE DATI
# ================================================================

@dataclass
class Impianto:
    """Singolo impianto pubblicitario OOH."""
    id: str
    tipo: str                         # tipo normalizzato: billboard|pensilina|metro|dooh
    tipo_originale: str               # valore grezzo da TipoImpianto in view_deepooh
    lat: float
    lon: float
    zona: str                         # codice istat comune (chiave per ZonaDemografica)
    formato: str            = "6x3"
    illuminato: bool        = True
    digitale: bool          = False
    angolo_visibilita: float   = 90.0
    velocita_media_kmh: float  = 30.0
    giorni_campagna: int    = 14


@dataclass
class ZonaDemografica:
    """Dati demografici per zona/comune (da view_deepooh + ISTAT)."""
    zona: str
    popolazione: int
    densita_abitativa: float
    eta_media: float
    pct_18_34: float
    pct_35_54: float
    pct_55plus: float
    auto_per_abitante: float         = 0.6
    flusso_pedonale_ora_peak: int    = 0
    flusso_veicolare_ora_peak: int   = 0
    ore_peak_giorno: float           = 6.5


@dataclass
class RisultatoImpianto:
    """Output della stima per un singolo impianto."""
    impianto_id: str
    ots_giornalieri: float
    ots_totali: float
    reach_univoco: float
    frequency_media: float
    grp: float
    copertura_pct: float
    profilo_demo: dict               = field(default_factory=dict)
    intervallo_confidenza_95: tuple  = (0.0, 0.0)
    note: list                       = field(default_factory=list)


# ================================================================
# MAPPING TIPO IMPIANTO — legge TipoImpianto da view_deepooh
# ================================================================

TIPO_MAP = {
    # ── Billboard / Poster ───────────────────────────────────────
    "billboard":          "billboard",
    "poster":             "billboard",
    "manifesto":          "billboard",
    "6x3":                "billboard",
    "murale":             "billboard",
    "maxi affissione":    "billboard",
    "super poster":       "billboard",
    "impianto speciale":  "billboard",

    # ── Pensilina / Shelter ──────────────────────────────────────
    "pensilina":          "pensilina",
    "shelter":            "pensilina",
    "fermata":            "pensilina",
    "fermata bus":        "pensilina",
    "tram":               "pensilina",
    "arredo urbano":      "pensilina",
    "totem":              "pensilina",

    # ── Metro / Stazione ─────────────────────────────────────────
    "metro":              "metro",
    "metropolitana":      "metro",
    "stazione":           "metro",
    "stazione fs":        "metro",
    "aeroporto":          "metro",
    "centro commerciale": "metro",

    # ── DOOH / Digitale ──────────────────────────────────────────
    "digital":            "dooh",
    "dooh":               "dooh",
    "led":                "dooh",
    "dynamic":            "dooh",
    "digitale":           "dooh",
    "schermo led":        "dooh",
    "video wall":         "dooh",
    "display digitale":   "dooh",
}


def normalizza_tipo(tipo_raw: str) -> str:
    """
    Converte il valore grezzo di TipoImpianto (da view_deepooh)
    nel tipo interno del modello.
    Strategia: corrispondenza esatta → sottostringa → fallback billboard.
    """
    if not tipo_raw:
        return "billboard"

    t = tipo_raw.lower().strip()

    if t in TIPO_MAP:
        return TIPO_MAP[t]

    for chiave, valore in TIPO_MAP.items():
        if chiave in t:
            return valore

    print(f"    [WARN] TipoImpianto non mappato: '{tipo_raw}' → default billboard")
    return "billboard"


# ================================================================
# AGENTI
# ================================================================

class BaseAgent:
    def __init__(self, nome: str, verboso: bool = False):
        self.nome    = nome
        self.verboso = verboso

    def log(self, msg: str):
        if self.verboso:
            print(f"    [{self.nome}] {msg}")


# ── Traffic Agent ────────────────────────────────────────────────

class TrafficAgent(BaseAgent):
    """
    Stima il flusso grezzo di persone/giorno davanti all'impianto.
    Modello lognormale con CV=15% per la variabilità giornaliera.
    """
    MOLTIPLICATORI_ZONA = {
        "centro":         1.8,
        "semicentro":     1.3,
        "periferia":      0.7,
        "stazione":       2.5,
        "aeroporto":      3.0,
        "commerciale":    1.6,
        "residenziale":   0.5,
    }
    ADJ_TIPO = {
        "metro":     {"ped": 1.5, "vei": 0.2},
        "pensilina": {"ped": 1.2, "vei": 1.0},
        "dooh":      {"ped": 1.1, "vei": 1.1},
        "billboard": {"ped": 1.0, "vei": 1.0},
    }

    def __init__(self, verboso: bool = False):
        super().__init__("TrafficAgent", verboso)

    def stima_flusso(self, imp: Impianto, zona: ZonaDemografica) -> dict:
        molt = self.MOLTIPLICATORI_ZONA.get(imp.zona, 1.0)
        adj  = self.ADJ_TIPO.get(imp.tipo, {"ped": 1.0, "vei": 1.0})

        fp = zona.flusso_pedonale_ora_peak  * zona.ore_peak_giorno * adj["ped"] * molt
        fv = zona.flusso_veicolare_ora_peak * zona.ore_peak_giorno * adj["vei"] * molt

        sigma_ln  = math.sqrt(math.log(1 + 0.15 ** 2))
        mu_ln_ped = math.log(max(fp, 1)) - sigma_ln ** 2 / 2
        mu_ln_vei = math.log(max(fv, 1)) - sigma_ln ** 2 / 2

        self.log(
            f"{imp.id} [{imp.tipo_originale}→{imp.tipo}]: "
            f"pedoni={fmt_num(fp)}/g  veicoli={fmt_num(fv)}/g"
        )

        return {
            "flusso_totale_medio": fp + fv * 1.4,
            "mu_ln_ped": mu_ln_ped,
            "mu_ln_vei": mu_ln_vei,
            "sigma_ln":  sigma_ln,
        }


# ── Format Agent ─────────────────────────────────────────────────

class FormatAgent(BaseAgent):
    """Calcola Visibility Index e Attention Factor per tipo di formato."""
    VIS_BASE   = {"billboard": 0.85, "pensilina": 0.78, "metro": 0.90, "dooh": 0.92}
    ATT_FACTOR = {"billboard": 0.40, "pensilina": 0.55, "metro": 0.65, "dooh": 0.50}

    def __init__(self, verboso: bool = False):
        super().__init__("FormatAgent", verboso)

    def calcola_coefficienti(self, imp: Impianto) -> dict:
        vis_base = self.VIS_BASE.get(imp.tipo, 0.75)
        af       = self.ATT_FACTOR.get(imp.tipo, 0.45)

        ang_adj  = min(imp.angolo_visibilita / 90.0, 1.0) ** 0.5
        dist_vis = max(imp.angolo_visibilita / 90 * 50, 10)
        vel_ms   = imp.velocita_media_kmh / 3.6
        t_exp    = dist_vis / max(vel_ms, 0.5)
        vel_adj  = min(t_exp / 1.5, 1.0)

        illum_b = 1.15 if imp.illuminato else 1.0
        dig_b   = 1.10 if imp.digitale   else 1.0

        vi = min(vis_base * ang_adj * vel_adj * illum_b * dig_b, 1.0)
        self.log(f"{imp.id}: VI={vi:.3f}  AF={af:.2f}")

        return {"visibility_index": vi, "attention_factor": af}


# ── Demo Agent ───────────────────────────────────────────────────

class DemoAgent(BaseAgent):
    """Profilo demografico dell'audience, aggiustato per tipo di impianto."""
    DEMO_ADJ = {
        "billboard": {"18_34": 1.0, "35_54": 1.0, "55plus": 0.9},
        "pensilina": {"18_34": 1.2, "35_54": 1.1, "55plus": 0.8},
        "metro":     {"18_34": 1.5, "35_54": 1.2, "55plus": 0.5},
        "dooh":      {"18_34": 1.3, "35_54": 1.0, "55plus": 0.7},
    }

    def __init__(self, verboso: bool = False):
        super().__init__("DemoAgent", verboso)

    def profilo_audience(
        self,
        imp: Impianto,
        zona: ZonaDemografica,
        target_eta: Optional[str] = None,
    ) -> dict:
        adj = self.DEMO_ADJ.get(imp.tipo, {"18_34": 1.0, "35_54": 1.0, "55plus": 1.0})
        raw = {
            "18_34":  zona.pct_18_34  * adj["18_34"],
            "35_54":  zona.pct_35_54  * adj["35_54"],
            "55plus": zona.pct_55plus * adj["55plus"],
        }
        tot     = sum(raw.values())
        profilo = {k: v / tot for k, v in raw.items()}
        quota   = profilo.get(target_eta, 1.0) if target_eta else 1.0

        self.log(f"{imp.id}: quota_target={quota:.2%}")
        return {"distribuzione": profilo, "quota_target": quota}


# ── Time Agent ───────────────────────────────────────────────────

class TimeAgent(BaseAgent):
    """Coefficiente temporale flat con peso per giorno della settimana."""
    PESO_GIORNO = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00,
                   4: 1.10, 5: 1.25, 6: 0.90}

    def __init__(self, verboso: bool = False):
        super().__init__("TimeAgent", verboso)

    def coefficiente_temporale(self, imp: Impianto) -> dict:
        giorni = imp.giorni_campagna
        n_sett = giorni // 7
        resto  = giorni % 7

        peso_sett  = sum(self.PESO_GIORNO.values()) / 7
        peso_resto = (
            sum(self.PESO_GIORNO[g % 7] for g in range(resto)) / resto
            if resto > 0 else peso_sett
        )
        ct = (n_sett * 7 * peso_sett + resto * peso_resto) / giorni

        self.log(f"{imp.id}: CT={ct:.3f}  giorni={giorni}")
        return {"coeff_temporale": ct, "giorni_campagna": giorni}


# ── Estimator Agent ──────────────────────────────────────────────

class EstimatorAgent(BaseAgent):
    """Calcola OTS, Reach, Frequency, GRP + IC 95% via Monte Carlo."""
    K_REACH = {"billboard": 0.55, "pensilina": 0.45, "metro": 0.65, "dooh": 0.50}

    def __init__(self, verboso: bool = False):
        super().__init__("EstimatorAgent", verboso)

    def stima(
        self,
        imp: Impianto,
        zona: ZonaDemografica,
        flusso: dict,
        coeff_f: dict,
        demo: dict,
        tempo: dict,
        target_eta: Optional[str] = None,
        n_sim: int = 1000,
    ) -> RisultatoImpianto:

        vi     = coeff_f["visibility_index"]
        af     = coeff_f["attention_factor"]
        ct     = tempo["coeff_temporale"]
        giorni = imp.giorni_campagna
        k      = self.K_REACH.get(imp.tipo, 0.50)
        pop_t  = zona.popolazione * demo["quota_target"]

        ots_g   = flusso["flusso_totale_medio"] * vi * af * ct
        ots_tot = ots_g * giorni
        reach   = pop_t * (1 - math.exp(-k * ots_tot / max(pop_t, 1)))
        freq    = ots_tot / max(reach, 1)
        cop     = reach  / max(pop_t, 1)
        grp     = cop * 100 * freq

        # Monte Carlo IC 95%
        rng      = np.random.default_rng(42)
        campioni = []
        for _ in range(n_sim):
            fp_s  = rng.lognormal(flusso["mu_ln_ped"], flusso["sigma_ln"])
            fv_s  = rng.lognormal(flusso["mu_ln_vei"], flusso["sigma_ln"])
            vi_s  = vi * rng.uniform(0.95, 1.05)
            ots_s = (fp_s + fv_s * 1.4) * vi_s * af * ct * giorni
            r_s   = pop_t * (1 - math.exp(-k * ots_s / max(pop_t, 1)))
            campioni.append(r_s)

        ic = tuple(np.percentile(campioni, [2.5, 97.5]))

        self.log(
            f"{imp.id}: OTS_g={fmt_num(ots_g)}  "
            f"reach={fmt_num(reach)}  freq={fmt_num(freq,1)}  GRP={fmt_num(grp,0)}"
        )

        return RisultatoImpianto(
            impianto_id              = imp.id,
            ots_giornalieri          = round(ots_g),
            ots_totali               = round(ots_tot),
            reach_univoco            = round(reach),
            frequency_media          = round(freq, 2),
            grp                      = round(grp, 1),
            copertura_pct            = round(cop * 100, 2),
            profilo_demo             = demo["distribuzione"],
            intervallo_confidenza_95 = (round(ic[0]), round(ic[1])),
            note                     = [f"target={target_eta or 'tutti'}"],
        )


# ── Validator Agent ──────────────────────────────────────────────

class ValidatorAgent(BaseAgent):
    """Sanity check statistico sui risultati."""

    def __init__(self, verboso: bool = False):
        super().__init__("ValidatorAgent", verboso)

    def valida(self, res: RisultatoImpianto, zona: ZonaDemografica) -> RisultatoImpianto:
        w = []

        if res.reach_univoco > zona.popolazione:
            w.append("⚠ Reach cappato a popolazione zona")
            res.reach_univoco = zona.popolazione

        if res.frequency_media > 30:
            w.append(f"⚠ Frequency alta ({fmt_num(res.frequency_media,1)}x): zona sovra-esposta")
        elif res.frequency_media < 1.5:
            w.append(f"⚠ Frequency bassa ({fmt_num(res.frequency_media,1)}x): valuta più impianti")

        lo, hi = res.intervallo_confidenza_95
        if lo > 0 and res.reach_univoco > 0:
            cv = (hi - lo) / (2 * res.reach_univoco)
            if cv > 0.30:
                w.append(f"ℹ IC ampio (CV={fmt_pct(cv*100,0)}): dati traffico incerti")

        if w:
            res.note.extend(w)
            self.log(f"{res.impianto_id}: {len(w)} avvisi")

        return res


# ── Orchestrator Agent ───────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """Coordina la pipeline e aggrega i risultati."""

    def __init__(self, verboso: bool = False):
        super().__init__("Orchestrator", verboso)
        self.traffic = TrafficAgent(verboso)
        self.fmt     = FormatAgent(verboso)
        self.demo    = DemoAgent(verboso)
        self.time    = TimeAgent(verboso)
        self.est     = EstimatorAgent(verboso)
        self.val     = ValidatorAgent(verboso)

    def esegui_campagna(
        self,
        impianti: list[Impianto],
        zone: dict[str, ZonaDemografica],
        target_eta: Optional[str] = None,
        n_sim: int = 1000,
    ) -> pd.DataFrame:

        self.log(f"Pipeline: {len(impianti)} impianti  target={target_eta or 'tutti'}")
        rows = []

        for imp in impianti:
            zona = zone.get(imp.zona)
            if zona is None:
                print(f"    [WARN] zona '{imp.zona}' non trovata per {imp.id}, skip")
                continue

            flusso = self.traffic.stima_flusso(imp, zona)
            coeff  = self.fmt.calcola_coefficienti(imp)
            demo   = self.demo.profilo_audience(imp, zona, target_eta)
            tempo  = self.time.coefficiente_temporale(imp)
            res    = self.est.stima(imp, zona, flusso, coeff, demo, tempo, target_eta, n_sim)
            res    = self.val.valida(res, zona)

            rows.append({
                "impianto_id":     res.impianto_id,
                "tipo_originale":  imp.tipo_originale,
                "tipo_modello":    imp.tipo,
                "zona_istat":      imp.zona,
                "lat":             imp.lat,
                "lon":             imp.lon,
                "formato":         imp.formato,
                "giorni":          imp.giorni_campagna,
                "ots_giornalieri": res.ots_giornalieri,
                "ots_totali":      res.ots_totali,
                "reach_univoco":   res.reach_univoco,
                "ic_low_95":       res.intervallo_confidenza_95[0],
                "ic_high_95":      res.intervallo_confidenza_95[1],
                "frequency_media": res.frequency_media,
                "grp":             res.grp,
                "copertura_pct":   res.copertura_pct,
                "pct_18_34":       round(res.profilo_demo.get("18_34",  0) * 100, 1),
                "pct_35_54":       round(res.profilo_demo.get("35_54",  0) * 100, 1),
                "pct_55plus":      round(res.profilo_demo.get("55plus", 0) * 100, 1),
                "avvisi":          " | ".join(res.note),
            })

        return pd.DataFrame(rows)

    def aggrega_campagna(
        self,
        df: pd.DataFrame,
        zone: dict[str, ZonaDemografica],
    ) -> dict:
        if df.empty:
            return {}

        ots_tot   = df["ots_totali"].sum()
        reach_cum = df["reach_univoco"].sum() * 0.70
        pop_media = sum(z.popolazione for z in zone.values()) / len(zone)
        cop       = reach_cum / max(pop_media, 1)
        freq      = ots_tot   / max(reach_cum, 1)
        grp       = cop * 100 * freq

        return {
            "n_impianti":          len(df),
            "ots_totali_campagna": int(ots_tot),
            "reach_cumulativo":    int(reach_cum),
            "copertura_pct":       round(cop * 100, 2),
            "frequency_media":     round(freq, 2),
            "grp_campagna":        round(grp, 1),
            "top_reach":           df.loc[df["reach_univoco"].idxmax(), "impianto_id"],
            "top_grp":             df.loc[df["grp"].idxmax(), "impianto_id"],
        }


# ================================================================
# MYSQL LOADER
# ================================================================

class MySQLLoader:
    """
    Carica impianti e zone dal database MySQL.
    Il tipo impianto viene letto da TipoImpianto in view_deepooh,
    fonte autorevole rispetto a TipoImpiantoDesc di campaigns.
    """

    def __init__(self, host, user, password, database, port=3306):
        try:
            import mysql.connector
            self.conn = mysql.connector.connect(
                host=host, user=user, password=password,
                database=database, port=port,
                connection_timeout=10,
            )
            self.cursor = self.conn.cursor(dictionary=True)
            print(f"[DB] Connessione a {host}/{database} OK")
        except Exception as e:
            raise ConnectionError(f"[DB] Connessione fallita: {e}")

    def close(self):
        self.cursor.close()
        self.conn.close()

    def carica_impianti(self, id_campagna: int, giorni: int = 14) -> list[Impianto]:
        """
        Legge impianti da campaigns e recupera TipoImpianto da view_deepooh
        tramite JOIN su istatComune + CodiceInpe.
        Beneficia dell'indice idx_campagna_istat su campaigns
        e idx_istat_abitanti su view_deepooh.
        """
        self.cursor.execute("""
            SELECT
                c.id,
                c.Formato,
                c.Latitudine,
                c.Longitudine,
                c.istatComune,
                c.idProvincia,
                c.CodiceInpe,
                v.TipoImpianto
            FROM campaigns c
            LEFT JOIN view_deepooh v
                ON  v.istat     = c.istatComune
                AND v.CodiceInpe = c.CodiceInpe
            WHERE c.idCampagna = %s
              AND c.Latitudine  IS NOT NULL
              AND c.Longitudine IS NOT NULL
            GROUP BY c.id
        """, (id_campagna,))

        rows     = self.cursor.fetchall()
        impianti = []

        for r in rows:
            try:
                lat = float(str(r["Latitudine"]).replace(",", ".").strip())
                lon = float(str(r["Longitudine"]).replace(",", ".").strip())
            except (ValueError, TypeError):
                continue

            tipo_raw = (r.get("TipoImpianto") or "").strip()
            tipo     = normalizza_tipo(tipo_raw)

            velocita = {"billboard": 40, "pensilina": 5,
                        "metro": 2, "dooh": 30}.get(tipo, 30)

            zona_key = r["istatComune"] or f"prov_{r['idProvincia']}"

            impianti.append(Impianto(
                id                 = str(r["id"]),
                tipo               = tipo,
                tipo_originale     = tipo_raw or "N/D",
                lat                = lat,
                lon                = lon,
                zona               = zona_key,
                formato            = r["Formato"] or "6x3",
                illuminato         = True,
                digitale           = (tipo == "dooh"),
                angolo_visibilita  = 90.0,
                velocita_media_kmh = float(velocita),
                giorni_campagna    = giorni,
            ))

        print(f"  [DB] Campagna {id_campagna}: {len(impianti)} impianti caricati")
        return impianti

    def carica_zone(self, id_campagna: int) -> dict[str, ZonaDemografica]:
        """
        Carica dati demografici da view_deepooh per i comuni
        degli impianti della campagna.
        Usa indice idx_istat_abitanti per performance ottimale.
        """
        self.cursor.execute("""
            SELECT DISTINCT
                v.istat,
                v.comune,
                v.Regione,
                v.TipoImpianto,
                v.abitanti,
                v.grpAdu,
                v.grpGravAdu
            FROM view_deepooh v
            INNER JOIN campaigns c
                ON v.istat = c.istatComune
            WHERE c.idCampagna = %s
              AND v.abitanti   IS NOT NULL
              AND v.abitanti   > 0
            limit 5
        """, (id_campagna,))

        rows = self.cursor.fetchall()
        zone = {}

        for r in rows:
            istat = (r["istat"] or "").strip()
            if not istat or istat in zone:
                continue

            abitanti = int(r["abitanti"] or 0)
            if abitanti == 0:
                continue

            grp_adu  = float(r["grpAdu"] or 0)
            densita  = self._stima_densita(abitanti)
            flusso_p = self._backsolve_flusso(grp_adu, abitanti)

            zone[istat] = ZonaDemografica(
                zona                      = istat,
                popolazione               = abitanti,
                densita_abitativa         = densita,
                eta_media                 = 42.0,
                pct_18_34                 = 0.28,
                pct_35_54                 = 0.37,
                pct_55plus                = 0.35,
                flusso_pedonale_ora_peak  = flusso_p,
                flusso_veicolare_ora_peak = int(flusso_p * 0.6),
                ore_peak_giorno           = 6.5,
            )

        print(f"  [DB] Campagna {id_campagna}: {len(zone)} zone caricate")
        return zone

    @staticmethod
    def _stima_densita(abitanti: int) -> float:
        if abitanti > 500000: return 7000.0
        if abitanti > 100000: return 3500.0
        if abitanti > 50000:  return 1500.0
        if abitanti > 10000:  return 600.0
        return 200.0

    @staticmethod
    def _backsolve_flusso(grp_adu: float, abitanti: int) -> int:
        """Stima flusso pedonale dal GRP certificato presente in view_deepooh."""
        if grp_adu <= 0 or abitanti <= 0:
            return max(int(abitanti * 0.03), 100)
        return max(int((grp_adu / 100) * abitanti / 14 / 6), 100)


# ================================================================
# STAMPA RIEPILOGO A CONSOLE (numeri in formato italiano)
# ================================================================

def stampa_riepilogo(id_camp: int, df: pd.DataFrame, agg: dict):
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  CAMPAGNA {id_camp}  —  {agg['n_impianti']} impianti  —  {GIORNI} giorni")
    print(sep)
    print(f"  OTS totali campagna  : {fmt_num(agg['ots_totali_campagna']):>16}")
    print(f"  Reach cumulativo     : {fmt_num(agg['reach_cumulativo']):>16}")
    print(f"  Copertura %          : {fmt_pct(agg['copertura_pct']):>15}")
    print(f"  Frequency media      : {fmt_num(agg['frequency_media'], 1):>15}x")
    print(f"  GRP campagna         : {fmt_num(agg['grp_campagna'], 0):>16}")
    print(f"  Top impianto reach   : {agg['top_reach']}")
    print(f"  Top impianto GRP     : {agg['top_grp']}")

    print(f"\n  {'ID':<12} {'Tipo DB':<22} {'Mod.':<10} "
          f"{'Reach':>11} {'IC−':>9} {'IC+':>9} {'Freq':>7} {'GRP':>9}")
    print(f"  {'─'*12} {'─'*22} {'─'*10} "
          f"{'─'*11} {'─'*9} {'─'*9} {'─'*7} {'─'*9}")

    for _, row in df.iterrows():
        print(
            f"  {str(row['impianto_id']):<12} "
            f"{str(row['tipo_originale']):<22} "
            f"{str(row['tipo_modello']):<10} "
            f"{fmt_num(row['reach_univoco']):>11} "
            f"{fmt_num(row['ic_low_95']):>9} "
            f"{fmt_num(row['ic_high_95']):>9} "
            f"{fmt_num(row['frequency_media'], 1):>7} "
            f"{fmt_num(row['grp'], 0):>9}"
        )

    avvisi = df[df["avvisi"].str.contains("⚠|ℹ", na=False)]
    if not avvisi.empty:
        print(f"\n  Avvisi validatore:")
        for _, row in avvisi.iterrows():
            for av in str(row["avvisi"]).split(" | "):
                if "⚠" in av or "ℹ" in av:
                    print(f"    {row['impianto_id']}: {av}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        OOH Campaign Modeler — Multi-Agent System            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Target eta  : {TARGET_ETA or 'tutti'}")
    print(f"  Durata      : {GIORNI} giorni (flat)")
    print(f"  Campagne    : {CAMPAGNE_TEST}")
    print(f"  Simulazioni : {N_SIMULAZIONI} (Monte Carlo IC 95%)")

    loader   = MySQLLoader(**DB_CONFIG)
    orc      = OrchestratorAgent(verboso=VERBOSO)
    tutti_df = []

    for id_camp in CAMPAGNE_TEST:
        print(f"\n▶  Elaborazione campagna {id_camp}...")

        impianti = loader.carica_impianti(id_campagna=id_camp, giorni=GIORNI)
        zone     = loader.carica_zone(id_campagna=id_camp)

        if not impianti:
            print(f"  [SKIP] Nessun impianto valido per campagna {id_camp}")
            continue
        if not zone:
            print(f"  [SKIP] Nessuna zona demografica per campagna {id_camp}")
            continue

        df  = orc.esegui_campagna(impianti, zone, target_eta=TARGET_ETA, n_sim=N_SIMULAZIONI)
        agg = orc.aggrega_campagna(df, zone)

        stampa_riepilogo(id_camp, df, agg)

        df["idCampagna"] = id_camp
        tutti_df.append(df)

    loader.close()

    # ── Export CSV formato italiano ───────────────────────────────
    if tutti_df:
        df_export = pd.concat(tutti_df, ignore_index=True)

        colonne = [
            "idCampagna", "impianto_id", "tipo_originale", "tipo_modello",
            "zona_istat", "lat", "lon", "formato", "giorni",
            "ots_giornalieri", "ots_totali",
            "reach_univoco", "ic_low_95", "ic_high_95",
            "frequency_media", "grp", "copertura_pct",
            "pct_18_34", "pct_35_54", "pct_55plus",
            "avvisi",
        ]
        df_export = df_export[[c for c in colonne if c in df_export.columns]]

        out_path = Path(__file__).parent / OUTPUT_CSV
        dataframe_to_csv_ita(df_export, out_path)

        print(f"\n{'═'*72}")
        print(f"  Campagne elaborate : {len(tutti_df)}")
        print(f"  Impianti totali    : {fmt_num(len(df_export))}")
        print(f"  File output        : {out_path}")
        print(f"  Formato CSV        : sep=;  decimali=,  migliaia=.  utf-8-sig")
        print(f"{'═'*72}\n")
    else:
        print("\n  Nessun risultato da esportare.")


if __name__ == "__main__":
    main()
