"""
population_builder.py — Genera la popolazione di agenti per un comune.

PROFILO_MODE:
    0 = solo Claude Haiku  (profili arricchiti, ~5 cent/campagna)
    1 = solo Python puro   (profili statistici, gratuito)
    2 = entrambi + analisi delle differenze e stima dell'errore
"""
from __future__ import annotations
import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .config import (
    DB_CONFIG, ANTHROPIC_API_KEY, CLAUDE_HAIKU_MODEL,
    SIM_SCALE, MIN_AGENTI_PER_COMUNE, MAX_AGENTI_PER_COMUNE,
    PROFILO_MODE, SIM_SEED,
)
from .istat_loader import DatiComuneISTAT
from .city_mapper import CityMap

log = logging.getLogger(__name__)


# ── Dataclass agente ──────────────────────────────────────────────────────────
@dataclass
class AgentProfile:
    agent_id:            str
    tipo:                str        # studente|lavoratore|pensionato|genitore|disoccupato
    eta:                 int
    fascia_eta:          str        # "0-17"|"18-34"|"35-54"|"55-64"|"65+"
    sesso:               str        # M|F
    mezzo_preferito:     str        # piedi|bici|bus|auto
    nodo_casa:           int        # nodo OSM
    ora_sveglia:         int        # 5-9
    ora_partenza:        int        # 6-10
    ora_rientro:         int        # 16-22
    ha_pranzo_fuori:     bool
    ha_attivita_serale:  bool
    tipo_attivita_serale: str       # svago|spesa|sport|nessuna
    source:              str        # "claude"|"python"
    archetipo_key:       str        # hash archetipo per raggruppamento


# ── Archetipi ─────────────────────────────────────────────────────────────────
# Chiave univoca che identifica un archetipo demografico
def _archetipo_key(tipo: str, fascia_eta: str, mezzo: str, sesso: str) -> str:
    raw = f"{tipo}|{fascia_eta}|{mezzo}|{sesso}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ── Distribuzione mezzi di trasporto ─────────────────────────────────────────
# Fonte: ISTAT Mobilità urbana 2022
_MEZZO_PER_TIPO: dict[str, dict[str, float]] = {
    "studente":    {"piedi": 0.30, "bici": 0.15, "bus": 0.40, "auto": 0.15},
    "lavoratore":  {"piedi": 0.15, "bici": 0.10, "bus": 0.30, "auto": 0.45},
    "pensionato":  {"piedi": 0.45, "bici": 0.10, "bus": 0.30, "auto": 0.15},
    "genitore":    {"piedi": 0.20, "bici": 0.08, "bus": 0.22, "auto": 0.50},
    "disoccupato": {"piedi": 0.40, "bici": 0.15, "bus": 0.35, "auto": 0.10},
}

# Orari tipici per tipo (media, sigma)
_ORARI: dict[str, dict] = {
    "studente":    {"sveglia": (7, 0.5), "partenza": (7, 0.5), "rientro": (14, 1.5)},
    "lavoratore":  {"sveglia": (6, 0.7), "partenza": (7, 0.7), "rientro": (18, 1.0)},
    "pensionato":  {"sveglia": (7, 1.0), "partenza": (9, 1.5), "rientro": (13, 1.5)},
    "genitore":    {"sveglia": (6, 0.5), "partenza": (7, 0.5), "rientro": (17, 1.0)},
    "disoccupato": {"sveglia": (8, 1.5), "partenza": (10, 2.0),"rientro": (15, 2.0)},
}

# Probabilità attività serale per tipo
_SERALE: dict[str, float] = {
    "studente": 0.55, "lavoratore": 0.40, "pensionato": 0.25,
    "genitore": 0.35, "disoccupato": 0.45,
}
_TIPO_SERALE: dict[str, list] = {
    "studente":    ["svago", "svago", "sport", "spesa"],
    "lavoratore":  ["spesa", "svago", "sport", "nessuna"],
    "pensionato":  ["spesa", "svago", "nessuna", "nessuna"],
    "genitore":    ["spesa", "sport", "svago", "nessuna"],
    "disoccupato": ["svago", "spesa", "nessuna", "nessuna"],
}


# ── Assegnazione tipo agente da fascia età ────────────────────────────────────
def _assegna_tipo(fascia: str, rng: np.random.Generator,
                  tasso_occ: float) -> str:
    if fascia == "0-17":
        return "studente"
    if fascia == "18-34":
        return rng.choice(
            ["studente", "lavoratore", "disoccupato"],
            p=[0.35, 0.55, 0.10]
        )
    if fascia == "35-54":
        p_lav = min(tasso_occ * 1.1, 0.90)
        p_oth = (1 - p_lav) / 2
        return rng.choice(
            ["lavoratore", "genitore", "disoccupato"],
            p=[p_lav - p_oth, p_oth * 2, p_oth]
        )
    if fascia == "55-64":
        return rng.choice(
            ["lavoratore", "pensionato", "genitore"],
            p=[0.45, 0.40, 0.15]
        )
    return "pensionato"   # 65+


def _eta_da_fascia(fascia: str, rng: np.random.Generator) -> int:
    limiti = {"0-17": (6,17), "18-34": (18,34), "35-54": (35,54),
              "55-64": (55,64), "65+": (65,85)}
    lo, hi = limiti.get(fascia, (25,55))
    return int(rng.integers(lo, hi + 1))


# ═════════════════════════════════════════════════════════════════════════════
# GENERATORE PYTHON PURO
# ═════════════════════════════════════════════════════════════════════════════
def _genera_profilo_python(
    agent_id: str,
    fascia: str,
    sesso: str,
    city_map: CityMap,
    tasso_occ: float,
    rng: np.random.Generator,
) -> AgentProfile:
    tipo  = _assegna_tipo(fascia, rng, tasso_occ)
    eta   = _eta_da_fascia(fascia, rng)
    mezzi = _MEZZO_PER_TIPO[tipo]
    mezzo = rng.choice(list(mezzi.keys()), p=list(mezzi.values()))

    orari = _ORARI[tipo]
    sveglia   = int(np.clip(rng.normal(*orari["sveglia"]),   5, 10))
    partenza  = int(np.clip(rng.normal(*orari["partenza"]),  sveglia, 11))
    rientro   = int(np.clip(rng.normal(*orari["rientro"]),   12, 23))

    ha_serale = rng.random() < _SERALE[tipo]
    tipo_ser  = rng.choice(_TIPO_SERALE[tipo]) if ha_serale else "nessuna"
    pranzo    = rng.random() < 0.35

    nodi_disponibili = city_map.grafo_walk.nodes()
    nodo_casa = rng.choice(list(nodi_disponibili))

    return AgentProfile(
        agent_id            = agent_id,
        tipo                = tipo,
        eta                 = eta,
        fascia_eta          = fascia,
        sesso               = sesso,
        mezzo_preferito     = mezzo,
        nodo_casa           = int(nodo_casa),
        ora_sveglia         = sveglia,
        ora_partenza        = partenza,
        ora_rientro         = rientro,
        ha_pranzo_fuori     = bool(pranzo),
        ha_attivita_serale  = bool(ha_serale),
        tipo_attivita_serale= tipo_ser,
        source              = "python",
        archetipo_key       = _archetipo_key(tipo, fascia, mezzo, sesso),
    )


# ═════════════════════════════════════════════════════════════════════════════
# GENERATORE CLAUDE HAIKU
# ═════════════════════════════════════════════════════════════════════════════

# Cache archetipi già generati da Claude (evita N chiamate API)
_claude_cache: dict[str, dict] = {}


def _genera_archetipo_claude(
    tipo: str, fascia: str, mezzo: str, sesso: str, citta: str
) -> dict:
    """Chiede a Claude Haiku la routine giornaliera per un archetipo."""
    key = _archetipo_key(tipo, fascia, mezzo, sesso)
    if key in _claude_cache:
        return _claude_cache[key]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""Sei un esperto di mobilità urbana italiana.
Genera la routine giornaliera tipica per questo profilo in formato JSON.

Profilo:
- Tipo: {tipo}
- Fascia età: {fascia} anni
- Sesso: {sesso}
- Mezzo preferito: {mezzo}
- Città: {citta}

Rispondi SOLO con JSON valido, nessun testo aggiuntivo:
{{
  "ora_sveglia": <int 5-9>,
  "ora_partenza": <int 6-11>,
  "ora_rientro": <int 12-23>,
  "ha_pranzo_fuori": <bool>,
  "ha_attivita_serale": <bool>,
  "tipo_attivita_serale": "<svago|spesa|sport|nessuna>",
  "note": "<breve descrizione comportamento>"
}}"""

        msg = client.messages.create(
            model      = CLAUDE_HAIKU_MODEL,
            max_tokens = 200,
            messages   = [{"role": "user", "content": prompt}],
        )
        risultato = json.loads(msg.content[0].text.strip())
        _claude_cache[key] = risultato
        log.info(f"Claude archetipo [{tipo}|{fascia}|{mezzo}|{sesso}]: {risultato.get('note','')}")
        return risultato

    except Exception as e:
        log.warning(f"Claude fallito per archetipo {key}: {e} — uso Python puro")
        return {}


def _genera_profilo_claude(
    agent_id: str,
    fascia: str,
    sesso: str,
    city_map: CityMap,
    tasso_occ: float,
    rng: np.random.Generator,
    nome_comune: str,
) -> AgentProfile:
    tipo  = _assegna_tipo(fascia, rng, tasso_occ)
    eta   = _eta_da_fascia(fascia, rng)
    mezzi = _MEZZO_PER_TIPO[tipo]
    mezzo = rng.choice(list(mezzi.keys()), p=list(mezzi.values()))

    # Chiedi a Claude la routine (1 chiamata per archetipo, poi cache)
    claude_data = _genera_archetipo_claude(tipo, fascia, mezzo, sesso, nome_comune)

    # Usa dati Claude se disponibili, altrimenti fallback Python
    orari  = _ORARI[tipo]
    sveglia  = claude_data.get("ora_sveglia",
                int(np.clip(rng.normal(*orari["sveglia"]), 5, 10)))
    partenza = claude_data.get("ora_partenza",
                int(np.clip(rng.normal(*orari["partenza"]), sveglia, 11)))
    rientro  = claude_data.get("ora_rientro",
                int(np.clip(rng.normal(*orari["rientro"]), 12, 23)))
    ha_serale= claude_data.get("ha_attivita_serale",
                rng.random() < _SERALE[tipo])
    tipo_ser = claude_data.get("tipo_attivita_serale",
                rng.choice(_TIPO_SERALE[tipo]) if ha_serale else "nessuna")
    pranzo   = claude_data.get("ha_pranzo_fuori", rng.random() < 0.35)

    # Aggiungi rumore gaussiano agli orari (ogni agente è diverso)
    sveglia  = int(np.clip(sveglia  + rng.integers(-1, 2), 5, 10))
    partenza = int(np.clip(partenza + rng.integers(-1, 2), sveglia, 11))
    rientro  = int(np.clip(rientro  + rng.integers(-1, 2), 12, 23))

    nodo_casa = rng.choice(list(city_map.grafo_walk.nodes()))

    return AgentProfile(
        agent_id            = agent_id,
        tipo                = tipo,
        eta                 = eta,
        fascia_eta          = fascia,
        sesso               = sesso,
        mezzo_preferito     = mezzo,
        nodo_casa           = int(nodo_casa),
        ora_sveglia         = sveglia,
        ora_partenza        = partenza,
        ora_rientro         = rientro,
        ha_pranzo_fuori     = bool(pranzo),
        ha_attivita_serale  = bool(ha_serale),
        tipo_attivita_serale= str(tipo_ser),
        source              = "claude",
        archetipo_key       = _archetipo_key(tipo, fascia, mezzo, sesso),
    )


# ═════════════════════════════════════════════════════════════════════════════
# ANALISI DIFFERENZE (mode 2)
# ═════════════════════════════════════════════════════════════════════════════
def analizza_differenze(
    agenti_python: list[AgentProfile],
    agenti_claude: list[AgentProfile],
) -> dict:
    """
    Confronta le distribuzioni di attributi tra i due metodi.
    Calcola differenza media, deviazione standard e stima dell'errore
    su ogni attributo numerico e categorico.
    """
    import pandas as pd
    from scipy import stats as scipy_stats

    df_py = pd.DataFrame([asdict(a) for a in agenti_python])
    df_cl = pd.DataFrame([asdict(a) for a in agenti_claude])

    risultati = {}

    # Attributi numerici
    for col in ["ora_sveglia", "ora_partenza", "ora_rientro", "eta"]:
        py_vals = df_py[col].values
        cl_vals = df_cl[col].values
        diff    = abs(py_vals.mean() - cl_vals.mean())
        # t-test per verificare se le distribuzioni sono statisticamente diverse
        try:
            t_stat, p_val = scipy_stats.ttest_ind(py_vals, cl_vals)
        except Exception:
            t_stat, p_val = 0.0, 1.0

        risultati[col] = {
            "python_media":  round(float(py_vals.mean()), 2),
            "claude_media":  round(float(cl_vals.mean()), 2),
            "diff_assoluta": round(float(diff), 2),
            "diff_pct":      round(float(diff / max(py_vals.mean(), 0.01) * 100), 1),
            "p_value":       round(float(p_val), 4),
            "significativo": bool(p_val < 0.05),
        }

    # Attributi categorici: distribuzione per categoria
    for col in ["tipo", "mezzo_preferito", "tipo_attivita_serale"]:
        py_dist = df_py[col].value_counts(normalize=True).to_dict()
        cl_dist = df_cl[col].value_counts(normalize=True).to_dict()
        cats    = set(py_dist) | set(cl_dist)
        max_diff= max(
            abs(py_dist.get(c, 0) - cl_dist.get(c, 0)) for c in cats
        )
        risultati[col] = {
            "python":      {k: round(v, 3) for k, v in py_dist.items()},
            "claude":      {k: round(v, 3) for k, v in cl_dist.items()},
            "max_diff_pct": round(max_diff * 100, 1),
        }

    # Attributi booleani
    for col in ["ha_pranzo_fuori", "ha_attivita_serale"]:
        py_m = float(df_py[col].mean())
        cl_m = float(df_cl[col].mean())
        risultati[col] = {
            "python_pct": round(py_m * 100, 1),
            "claude_pct": round(cl_m * 100, 1),
            "diff_pct":   round(abs(py_m - cl_m) * 100, 1),
        }

    # Indice di somiglianza globale (0=identici, 1=massima differenza)
    diffs_num = [v["diff_pct"] for k, v in risultati.items()
                 if "diff_pct" in v and isinstance(v["diff_pct"], float)]
    errore_medio = round(sum(diffs_num) / max(len(diffs_num), 1), 2)

    risultati["_sommario"] = {
        "n_agenti_python": len(agenti_python),
        "n_agenti_claude": len(agenti_claude),
        "errore_medio_pct": errore_medio,
        "valutazione": (
            "EQUIVALENTI"   if errore_medio < 5  else
            "SIMILI"        if errore_medio < 15 else
            "DIFFERENZE RILEVANTI"
        ),
    }

    return risultati


def stampa_analisi(analisi: dict):
    """Stampa l'analisi differenze in modo leggibile."""
    s = analisi.get("_sommario", {})
    sep = "=" * 60
    print(f"\n{sep}")
    print("  ANALISI DIFFERENZE: Claude vs Python puro")
    print(sep)
    print(f"  Agenti Python : {s.get('n_agenti_python')}")
    print(f"  Agenti Claude : {s.get('n_agenti_claude')}")
    print(f"  Errore medio  : {s.get('errore_medio_pct')}%")
    print(f"  Valutazione   : {s.get('valutazione')}")
    print()

    print("  Attributi numerici:")
    for col in ["ora_sveglia", "ora_partenza", "ora_rientro", "eta"]:
        if col not in analisi:
            continue
        v = analisi[col]
        flag = " ← DIFFERENZA SIGNIFICATIVA" if v["significativo"] else ""
        print(f"    {col:<20} Python={v['python_media']:5.1f}  "
              f"Claude={v['claude_media']:5.1f}  "
              f"diff={v['diff_pct']:4.1f}%  p={v['p_value']:.3f}{flag}")

    print()
    print("  Distribuzione mezzo di trasporto:")
    if "mezzo_preferito" in analisi:
        m = analisi["mezzo_preferito"]
        tutti = set(m["python"]) | set(m["claude"])
        for k in sorted(tutti):
            py = m["python"].get(k, 0)
            cl = m["claude"].get(k, 0)
            print(f"    {k:<8} Python={py:.1%}  Claude={cl:.1%}  diff={abs(py-cl):.1%}")
        print(f"    max_diff={m['max_diff_pct']}%")

    print()
    print("  Attività serale e pranzo:")
    for col in ["ha_pranzo_fuori", "ha_attivita_serale"]:
        if col not in analisi:
            continue
        v = analisi[col]
        print(f"    {col:<22} Python={v['python_pct']}%  "
              f"Claude={v['claude_pct']}%  diff={v['diff_pct']}%")
    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════
def build_population(
    dati_istat: DatiComuneISTAT,
    city_map: CityMap,
    mode: int = None,
    seed: int = SIM_SEED,
) -> tuple[list[AgentProfile], Optional[dict]]:
    """
    Genera la popolazione di agenti per un comune.

    Args:
        dati_istat : dati demografici ISTAT del comune
        city_map   : grafo stradale OSM del comune
        mode       : 0=Claude, 1=Python, 2=entrambi (default da config)
        seed       : seed per riproducibilità

    Returns:
        (agenti, analisi)
        - agenti  : lista AgentProfile (fonte dipende da mode)
        - analisi : dict con analisi differenze (solo mode=2, altrimenti None)
    """
    if mode is None:
        mode = PROFILO_MODE

    # Calcola numero agenti
    n = int(dati_istat.popolazione_totale * SIM_SCALE)
    n = max(MIN_AGENTI_PER_COMUNE, min(MAX_AGENTI_PER_COMUNE, n))
    log.info(f"Generazione {n} agenti per {dati_istat.nome_comune} (mode={mode})")

    rng = np.random.default_rng(seed)

    # Campionamento demografico da distribuzioni ISTAT
    fasce  = list(dati_istat.distribuzione_eta.keys())
    pesi_f = list(dati_istat.distribuzione_eta.values())
    sessi  = list(dati_istat.distribuzione_sesso.keys())
    pesi_s = list(dati_istat.distribuzione_sesso.values())

    fasce_campionate = rng.choice(fasce, size=n, p=pesi_f)
    sessi_campionati = rng.choice(sessi, size=n, p=pesi_s)

    # ── Mode 1: solo Python ──────────────────────────────────────────────────
    if mode == 1:
        agenti = [
            _genera_profilo_python(
                f"{dati_istat.codice_istat}_{i:05d}",
                fasce_campionate[i], sessi_campionati[i],
                city_map, dati_istat.tasso_occupazione, rng,
            )
            for i in range(n)
        ]
        return agenti, None

    # ── Mode 0: solo Claude ──────────────────────────────────────────────────
    if mode == 0:
        if not ANTHROPIC_API_KEY:
            log.error("ANTHROPIC_API_KEY mancante — fallback a Python puro")
            return build_population(dati_istat, city_map, mode=1, seed=seed)

        agenti = []
        for i in range(n):
            a = _genera_profilo_claude(
                f"{dati_istat.codice_istat}_{i:05d}",
                fasce_campionate[i], sessi_campionati[i],
                city_map, dati_istat.tasso_occupazione, rng,
                dati_istat.nome_comune,
            )
            agenti.append(a)
            if (i + 1) % 100 == 0:
                log.info(f"  Generati {i+1}/{n} agenti (Claude)")
        return agenti, None

    # ── Mode 2: entrambi + analisi ───────────────────────────────────────────
    log.info("Mode 2: genero popolazione con ENTRAMBI i metodi...")

    # Python puro (seed fisso per confronto equo)
    rng_py = np.random.default_rng(seed)
    f_py   = rng_py.choice(fasce, size=n, p=pesi_f)
    s_py   = rng_py.choice(sessi, size=n, p=pesi_s)
    agenti_python = [
        _genera_profilo_python(
            f"{dati_istat.codice_istat}_PY_{i:05d}",
            f_py[i], s_py[i], city_map,
            dati_istat.tasso_occupazione, rng_py,
        )
        for i in range(n)
    ]

    # Claude (stesso seed → stessa demografia, solo routine diversa)
    if ANTHROPIC_API_KEY:
        rng_cl = np.random.default_rng(seed)
        f_cl   = rng_cl.choice(fasce, size=n, p=pesi_f)
        s_cl   = rng_cl.choice(sessi, size=n, p=pesi_s)
        agenti_claude = []
        for i in range(n):
            a = _genera_profilo_claude(
                f"{dati_istat.codice_istat}_CL_{i:05d}",
                f_cl[i], s_cl[i], city_map,
                dati_istat.tasso_occupazione, rng_cl,
                dati_istat.nome_comune,
            )
            agenti_claude.append(a)
        log.info(f"  Claude: {len(_claude_cache)} archetipi generati (API calls)")
    else:
        log.warning("ANTHROPIC_API_KEY mancante — analisi solo Python vs Python (seed diversi)")
        rng_cl2 = np.random.default_rng(seed + 1)
        f_cl2   = rng_cl2.choice(fasce, size=n, p=pesi_f)
        s_cl2   = rng_cl2.choice(sessi, size=n, p=pesi_s)
        agenti_claude = [
            _genera_profilo_python(
                f"{dati_istat.codice_istat}_CL_{i:05d}",
                f_cl2[i], s_cl2[i], city_map,
                dati_istat.tasso_occupazione, rng_cl2,
            )
            for i in range(n)
        ]

    # Analisi differenze
    try:
        analisi = analizza_differenze(agenti_python, agenti_claude)
        stampa_analisi(analisi)
    except ImportError:
        log.warning("scipy non installata — analisi statistica non disponibile")
        analisi = {"errore": "scipy non installata"}

    # Agenti finali: merge (Python come base, Claude come variante)
    # Per la simulazione si usano gli agenti Python come default
    # (disponibili sempre, anche senza API key)
    return agenti_python, analisi


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from .istat_loader import load_istat_data
    from .city_mapper import load_city_map

    dati  = load_istat_data(["017174"])
    d     = dati["017174"]
    cm    = load_city_map("017174", d.nome_comune, *d.coordinate_centroide, dist_m=3000)

    agenti, analisi = build_population(d, cm)
    print(f"\nAgenti generati: {len(agenti)}")
    print(f"Sorgente: {agenti[0].source}")
    for a in agenti[:3]:
        print(f"  {a.agent_id} | {a.tipo:12} | {a.fascia_eta} | "
              f"{a.mezzo_preferito:6} | sveglia={a.ora_sveglia} "
              f"rientro={a.ora_rientro} | serale={a.ha_attivita_serale}")
