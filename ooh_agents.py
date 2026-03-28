"""
OOH Campaign Modeler — Multi-Agent System
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class Impianto:
    id: str
    tipo: str
    lat: float
    lon: float
    zona: str
    formato: str = "6x3"
    illuminato: bool = True
    digitale: bool = False
    angolo_visibilita: float = 90.0
    altezza_m: float = 3.0
    velocita_media_kmh: float = 30.0
    giorni_campagna: int = 14

@dataclass
class ZonaDemografica:
    zona: str
    popolazione: int
    densita_abitativa: float
    eta_media: float
    pct_18_34: float
    pct_35_54: float
    pct_55plus: float
    auto_per_abitante: float = 0.6
    flusso_pedonale_ora_peak: int = 0
    flusso_veicolare_ora_peak: int = 0
    ore_peak_giorno: float = 6.0

@dataclass
class RisultatoImpianto:
    impianto_id: str
    ots_giornalieri: float
    ots_totali: float
    reach_univoco: float
    frequency_media: float
    grp: float
    copertura_pct: float
    profilo_demo: dict = field(default_factory=dict)
    intervallo_confidenza_95: tuple = (0.0, 0.0)
    note: list = field(default_factory=list)

class BaseAgent:
    def __init__(self, nome, verboso=False):
        self.nome = nome
        self.verboso = verboso
    def log(self, msg):
        if self.verboso:
            print(f"  [{self.nome}] {msg}")

class TrafficAgent(BaseAgent):
    MOLTIPLICATORI_ZONA = {
        "centro": 1.8, "semicentro": 1.3, "periferia": 0.7,
        "stazione": 2.5, "aeroporto": 3.0, "commerciale": 1.6, "residenziale": 0.5,
    }
    def __init__(self, verboso=False):
        super().__init__("TrafficAgent", verboso)

    def stima_flusso(self, imp, zona):
        molt = self.MOLTIPLICATORI_ZONA.get(imp.zona, 1.0)
        ped_base = zona.flusso_pedonale_ora_peak * zona.ore_peak_giorno
        vei_base = zona.flusso_veicolare_ora_peak * zona.ore_peak_giorno
        if imp.tipo == "metro":
            ped_base *= 1.5; vei_base *= 0.2
        elif imp.tipo == "pensilina":
            ped_base *= 1.2
        elif imp.tipo == "dooh":
            ped_base *= 1.1; vei_base *= 1.1
        fp = ped_base * molt
        fv = vei_base * molt
        sigma_ln = math.sqrt(math.log(1 + 0.15**2))
        mu_ln_ped = math.log(max(fp, 1)) - sigma_ln**2 / 2
        mu_ln_vei = math.log(max(fv, 1)) - sigma_ln**2 / 2
        self.log(f"{imp.id}: pedoni={fp:.0f}/g, veicoli={fv:.0f}/g")
        return {
            "pedonale_medio": fp, "veicolare_medio": fv,
            "flusso_totale_medio": fp + fv * 1.4,
            "mu_ln_ped": mu_ln_ped, "mu_ln_vei": mu_ln_vei, "sigma_ln": sigma_ln,
        }

class FormatAgent(BaseAgent):
    VIS_BASE = {"billboard": 0.85, "pensilina": 0.78, "metro": 0.90, "dooh": 0.92}
    ATT_FACTOR = {"billboard": 0.40, "pensilina": 0.55, "metro": 0.65, "dooh": 0.50}
    def __init__(self, verboso=False):
        super().__init__("FormatAgent", verboso)

    def calcola_coefficienti(self, imp):
        vis_base = self.VIS_BASE.get(imp.tipo, 0.75)
        af = self.ATT_FACTOR.get(imp.tipo, 0.45)
        ang_adj = min(imp.angolo_visibilita / 90.0, 1.0) ** 0.5
        dist_vis = max(imp.angolo_visibilita / 90 * 50, 10)
        vel_ms = imp.velocita_media_kmh / 3.6
        t_exp = dist_vis / max(vel_ms, 0.5)
        vel_adj = min(t_exp / 1.5, 1.0)
        illum_b = 1.15 if imp.illuminato else 1.0
        dig_b = 1.10 if imp.digitale else 1.0
        vi = min(vis_base * ang_adj * vel_adj * illum_b * dig_b, 1.0)
        self.log(f"{imp.id}: VI={vi:.3f}")
        return {"visibility_index": vi, "attention_factor": af,
                "illum_bonus": illum_b, "digital_bonus": dig_b}

class DemoAgent(BaseAgent):
    DEMO_ADJ = {
        "billboard": {"18_34": 1.0, "35_54": 1.0, "55plus": 0.9},
        "pensilina":  {"18_34": 1.2, "35_54": 1.1, "55plus": 0.8},
        "metro":      {"18_34": 1.5, "35_54": 1.2, "55plus": 0.5},
        "dooh":       {"18_34": 1.3, "35_54": 1.0, "55plus": 0.7},
    }
    def __init__(self, verboso=False):
        super().__init__("DemoAgent", verboso)

    def profilo_audience(self, imp, zona, target_eta=None):
        adj = self.DEMO_ADJ.get(imp.tipo, {"18_34": 1.0, "35_54": 1.0, "55plus": 1.0})
        raw = {
            "18_34": zona.pct_18_34 * adj["18_34"],
            "35_54": zona.pct_35_54 * adj["35_54"],
            "55plus": zona.pct_55plus * adj["55plus"],
        }
        tot = sum(raw.values())
        profilo = {k: v / tot for k, v in raw.items()}
        quota_target = profilo.get(target_eta, 1.0) if target_eta else 1.0
        self.log(f"{imp.id}: quota_target={quota_target:.2%}")
        return {"distribuzione": profilo, "quota_target": quota_target, "eta_media_zona": zona.eta_media}

class TimeAgent(BaseAgent):
    PESO_GIORNO = {0:1.0,1:1.0,2:1.0,3:1.0,4:1.1,5:1.25,6:0.90}
    def __init__(self, verboso=False):
        super().__init__("TimeAgent", verboso)

    def coefficiente_temporale(self, imp):
        giorni = imp.giorni_campagna
        n_sett = giorni // 7; resto = giorni % 7
        peso_sett = sum(self.PESO_GIORNO.values()) / 7
        peso_resto = sum(self.PESO_GIORNO[g % 7] for g in range(resto)) / max(resto, 1)
        ct = (n_sett * 7 * peso_sett + resto * peso_resto) / giorni
        ore_utili = 18 if imp.illuminato else 12
        self.log(f"{imp.id}: coeff={ct:.3f}, ore={ore_utili}")
        return {"coeff_temporale": ct, "ore_utili_giorno": ore_utili, "giorni_campagna": giorni}

class EstimatorAgent(BaseAgent):
    K_REACH = {"billboard": 0.55, "pensilina": 0.45, "metro": 0.65, "dooh": 0.50}
    def __init__(self, verboso=False):
        super().__init__("EstimatorAgent", verboso)

    def stima(self, imp, zona, flusso, coeff_f, demo, tempo, target_eta=None, n_sim=1000):
        vi = coeff_f["visibility_index"]; af = coeff_f["attention_factor"]
        ct = tempo["coeff_temporale"]; giorni = imp.giorni_campagna
        pop = zona.popolazione; quota = demo["quota_target"]
        ots_g = flusso["flusso_totale_medio"] * vi * af * ct
        ots_tot = ots_g * giorni
        k = self.K_REACH.get(imp.tipo, 0.50)
        pop_t = pop * quota
        reach = pop_t * (1 - math.exp(-k * ots_tot / max(pop_t, 1)))
        freq = ots_tot / max(reach, 1)
        cop = reach / max(pop_t, 1)
        grp = cop * 100 * freq
        # Monte Carlo IC 95%
        rng = np.random.default_rng(42)
        campioni = []
        for _ in range(n_sim):
            fp = rng.lognormal(flusso["mu_ln_ped"], flusso["sigma_ln"])
            fv = rng.lognormal(flusso["mu_ln_vei"], flusso["sigma_ln"])
            fs = fp + fv * 1.4
            vi_s = vi * rng.uniform(0.95, 1.05)
            ots_s = fs * vi_s * af * ct * giorni
            r_s = pop_t * (1 - math.exp(-k * ots_s / max(pop_t, 1)))
            campioni.append(r_s)
        ic = tuple(np.percentile(campioni, [2.5, 97.5]))
        self.log(f"{imp.id}: OTS_g={ots_g:.0f} reach={reach:.0f} freq={freq:.1f} GRP={grp:.1f}")
        return RisultatoImpianto(
            imp.id, round(ots_g), round(ots_tot), round(reach), round(freq, 2),
            round(grp, 1), round(cop * 100, 2), demo["distribuzione"],
            (round(ic[0]), round(ic[1])), [f"target={target_eta or 'tutti'}"]
        )

class ValidatorAgent(BaseAgent):
    def __init__(self, verboso=False):
        super().__init__("ValidatorAgent", verboso)

    def valida(self, res, zona, imp):
        w = []
        if res.reach_univoco > zona.popolazione:
            w.append("⚠ Reach cappato a popolazione zona")
            res.reach_univoco = zona.popolazione
        if res.frequency_media > 30:
            w.append(f"⚠ Frequency alta ({res.frequency_media:.1f}x)")
        if res.frequency_media < 1.5:
            w.append(f"⚠ Frequency bassa ({res.frequency_media:.1f}x)")
        lo, hi = res.intervallo_confidenza_95
        if lo > 0 and (hi - lo) / (2 * res.reach_univoco) > 0.30:
            w.append("ℹ IC ampio: dati traffico incerti")
        if w:
            res.note.extend(w)
        return res

class OrchestratorAgent(BaseAgent):
    def __init__(self, verboso=False):
        super().__init__("Orchestrator", verboso)
        self.traffic = TrafficAgent(verboso)
        self.fmt     = FormatAgent(verboso)
        self.demo    = DemoAgent(verboso)
        self.time    = TimeAgent(verboso)
        self.est     = EstimatorAgent(verboso)
        self.val     = ValidatorAgent(verboso)

    def esegui_campagna(self, impianti, zone, target_eta=None, n_sim=1000):
        self.log(f"Pipeline: {len(impianti)} impianti, target={target_eta or 'tutti'}")
        rows = []
        for imp in impianti:
            zona = zone.get(imp.zona)
            if zona is None: continue
            flusso = self.traffic.stima_flusso(imp, zona)
            coeff  = self.fmt.calcola_coefficienti(imp)
            demo   = self.demo.profilo_audience(imp, zona, target_eta)
            tempo  = self.time.coefficiente_temporale(imp)
            res    = self.est.stima(imp, zona, flusso, coeff, demo, tempo, target_eta, n_sim)
            res    = self.val.valida(res, zona, imp)
            rows.append({
                "impianto_id": res.impianto_id,
                "tipo": imp.tipo,
                "zona": imp.zona,
                "ots_giornalieri": res.ots_giornalieri,
                "ots_totali": res.ots_totali,
                "reach_univoco": res.reach_univoco,
                "frequency_media": res.frequency_media,
                "grp": res.grp,
                "copertura_pct": res.copertura_pct,
                "ic_low_95": res.intervallo_confidenza_95[0],
                "ic_high_95": res.intervallo_confidenza_95[1],
                "pct_18_34": res.profilo_demo.get("18_34", 0),
                "avvisi": " | ".join(res.note),
            })
        return pd.DataFrame(rows)

    def aggrega_campagna(self, df, zone):
        if df.empty: return {}
        ots_tot = df["ots_totali"].sum()
        reach_cum = df["reach_univoco"].sum() * 0.70
        pop_media = sum(z.popolazione for z in zone.values()) / len(zone)
        cop = reach_cum / max(pop_media, 1)
        freq = ots_tot / max(reach_cum, 1)
        grp = cop * 100 * freq
        return {
            "n_impianti": len(df),
            "ots_totali_campagna": int(ots_tot),
            "reach_cumulativo": int(reach_cum),
            "copertura_pct": round(cop * 100, 2),
            "frequency_media": round(freq, 2),
            "grp_campagna": round(grp, 1),
            "top_reach": df.loc[df["reach_univoco"].idxmax(), "impianto_id"],
            "top_grp":   df.loc[df["grp"].idxmax(), "impianto_id"],
        }

# ── Dati esempio ──────────────────────────────
def crea_dati_esempio():
    zone = {
        "centro": ZonaDemografica("centro", 85000, 12000, 42, 0.32, 0.38, 0.30,
                                  0.45, 3200, 1800, 7.0),
        "stazione": ZonaDemografica("stazione", 40000, 8000, 35, 0.45, 0.35, 0.20,
                                    0.30, 5500, 2200, 8.0),
        "periferia": ZonaDemografica("periferia", 120000, 3500, 46, 0.22, 0.40, 0.38,
                                     0.75, 800, 3500, 5.5),
        "commerciale": ZonaDemografica("commerciale", 30000, 6000, 38, 0.40, 0.38, 0.22,
                                       0.55, 2800, 2000, 9.0),
    }
    impianti = [
        Impianto("BB-001", "billboard", 45.07, 7.68, "centro",
                 illuminato=True, angolo_visibilita=110, velocita_media_kmh=25, giorni_campagna=14),
        Impianto("BB-002", "billboard", 45.06, 7.70, "periferia",
                 illuminato=True, angolo_visibilita=90,  velocita_media_kmh=50, giorni_campagna=14),
        Impianto("PEN-001", "pensilina", 45.07, 7.67, "centro",
                 illuminato=True, angolo_visibilita=80,  velocita_media_kmh=5,  giorni_campagna=14),
        Impianto("PEN-002", "pensilina", 45.06, 7.69, "stazione",
                 illuminato=True, angolo_visibilita=75,  velocita_media_kmh=4,  giorni_campagna=14),
        Impianto("MTR-001", "metro",     45.06, 7.68, "stazione",
                 illuminato=True, angolo_visibilita=120, velocita_media_kmh=2,  giorni_campagna=14),
        Impianto("DOH-001", "dooh",      45.07, 7.69, "commerciale",
                 illuminato=True, digitale=True, angolo_visibilita=100, velocita_media_kmh=20, giorni_campagna=14),
    ]
    return impianti, zone

if __name__ == "__main__":
    print("=" * 65)
    print("  OOH Campaign Modeler — Multi-Agent System")
    print("=" * 65)
    impianti, zone = crea_dati_esempio()
    orc = OrchestratorAgent(verboso=True)
    print("\n▶  Pipeline (target: 18-34 anni)\n")
    df = orc.esegui_campagna(impianti, zone, target_eta="18_34", n_sim=1000)
    agg = orc.aggrega_campagna(df, zone)

    print("\n" + "─" * 65)
    print("RISULTATI PER IMPIANTO")
    print("─" * 65)
    pd.set_option("display.float_format", "{:,.1f}".format)
    cols = ["impianto_id","tipo","zona","ots_totali","reach_univoco","frequency_media","grp","copertura_pct"]
    print(df[cols].to_string(index=False))

    print("\n" + "─" * 65)
    print("INTERVALLI DI CONFIDENZA 95%")
    print("─" * 65)
    print(df[["impianto_id","reach_univoco","ic_low_95","ic_high_95"]].to_string(index=False))

    print("\n" + "─" * 65)
    print("METRICHE AGGREGATE CAMPAGNA")
    print("─" * 65)
    for k, v in agg.items():
        print(f"  {k:<28}: {v}")

    df.to_csv("risultati.csv",sep=";", index=False)
    print("\nExport → risultati.csv ✓")
    print("=" * 65)
