"""
config.py — Parametri globali del sistema ABM OOH.
Legge dal file .env nella root del progetto.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica .env dalla root del progetto (una cartella sopra abm/)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "dbooh"),
}

# ── Claude API ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ── Modalità generazione profili agenti ──────────────────────────────────────
# 0 = solo Claude Haiku
# 1 = solo Python puro (statistico)
# 2 = entrambi + analisi differenze
PROFILO_MODE = int(os.getenv("PROFILO_MODE", 2))

# ── Simulazione ───────────────────────────────────────────────────────────────
# SIM_SCALE: frazione della popolazione reale da simulare
# 0.01 = rapida, 0.05 = buona, 0.10 = ottima
SIM_SCALE   = float(os.getenv("SIM_SCALE", 0.05))
SIM_GIORNI  = int(os.getenv("SIM_GIORNI", 14))
SIM_WORKERS = int(os.getenv("SIM_WORKERS", 4))
SIM_SEED    = int(os.getenv("SIM_SEED", 42))

# Numero minimo e massimo di agenti per comune (override di SIM_SCALE)
MIN_AGENTI_PER_COMUNE = 200
MAX_AGENTI_PER_COMUNE = 5000

# ── Visibilità impianti ───────────────────────────────────────────────────────
# Raggio base in metri; raffinato per mezzo di trasporto
RAGGIO_DEFAULT_M = 50

RAGGIO_PER_MEZZO: dict[str, int] = {
    "piedi": 30,
    "bici":  40,
    "bus":   50,
    "auto":  80,
}

# Solo impianti illuminati sono visibili in fascia notturna
ORA_NOTTURNA_INIZIO = 21   # 21:00
ORA_NOTTURNA_FINE   = 7    # 07:00

# ── Timeline giornata ─────────────────────────────────────────────────────────
SLOT_MINUTI = 15           # granularità simulazione (minuti)
ORE_GIORNATA = 24
SLOT_PER_GIORNO = (ORE_GIORNATA * 60) // SLOT_MINUTI   # 96 slot

# ── Tipi impianto OOH riconosciuti ───────────────────────────────────────────
TIPO_MAP: dict[str, str] = {
    "arredo":       "arredo",
    "comunal":      "comunali",
    "digital":      "digital",
    "dooh":         "digital",
    "led":          "digital",
    "dinamica":     "dinamica",
    "dynamic":      "dinamica",
    "medi":         "medi",
    "medio":        "medi",
    "poster":       "poster",
    "manifesto":    "poster",
    "billboard":    "poster",
    "speciale":     "speciale",
    "special":      "speciale",
    "metro":        "metro",
    "metropolitana":"metro",
    "stazione":     "metro",
}

# ── Percorsi cache ────────────────────────────────────────────────────────────
CACHE_DIR      = _ROOT / "cache"
OSM_CACHE_DIR  = CACHE_DIR / "osm"
ISTAT_CACHE_DIR= CACHE_DIR / "istat"

# ── Parametri demografici di fallback (se ISTAT non disponibile) ──────────────
DEMO_FALLBACK = {
    "distribuzione_eta": {
        "0-17":  0.16,
        "18-34": 0.22,
        "35-54": 0.30,
        "55-64": 0.14,
        "65+":   0.18,
    },
    "distribuzione_sesso": {"M": 0.485, "F": 0.515},
    "tasso_occupazione":   0.58,
    "tasso_pendolarismo":  0.45,
    "auto_per_abitante":   0.62,
}
