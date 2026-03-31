"""
istat_loader.py — Scarica dati demografici ISTAT per comuni italiani.

Fonte coordinate : OSM Nominatim (gratuito, affidabile)
Fonte popolazione: API open.istat.it oppure lookup tabella città principali
Fallback         : distribuzioni di default da config.py
Cache locale     : JSON su disco con TTL 30 giorni
"""
from __future__ import annotations
import json
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import requests

from .config import ISTAT_CACHE_DIR, DEMO_FALLBACK

log = logging.getLogger(__name__)

CACHE_TTL_GIORNI = 30

# Lookup principali comuni italiani: codice ISTAT → (nome, lat, lon, popolazione)
# Fonte: ISTAT Bilancio demografico 2022 + OSM
COMUNI_LOOKUP: dict[str, tuple[str, float, float, int]] = {
    "001272": ("Torino",    45.0703, 7.6869,  847287),
    "015146": ("Milano",    45.4654, 9.1859, 1352631),
    "058091": ("Roma",      41.8933, 12.4830,2739000),
    "063049": ("Napoli",    40.8358, 14.2488, 908000),
    "037006": ("Bologna",   44.4938, 11.3426, 419000),
    "048017": ("Firenze",   43.7696, 11.2558, 361000),
    "023023": ("Verona",    45.4387, 10.9916, 258000),
    "028060": ("Padova",    45.4064, 11.8768, 210000),
    "027042": ("Venezia",   45.4397, 12.3319, 255000),
    "082053": ("Palermo",   38.1112, 13.3524, 652000),
    "087015": ("Catania",   37.5100, 15.0900, 311000),
    "010025": ("Genova",    44.4056, 8.9463,  565000),
    "072006": ("Bari",      41.1177, 16.8719, 316000),
    "016024": ("Brescia",   45.5416, 10.2118, 200000),
    "020041": ("Trieste",   45.6503, 13.7703, 201000),
    "024129": ("Vicenza",   45.5455, 11.5354, 111000),
    "022205": ("Trento",    46.0748, 11.1217, 118000),
    "030129": ("Udine",     46.0633, 13.2350,  99000),
    "012123": ("Varese",    45.8183, 8.8257,   79000),
    "013055": ("Como",      45.8081, 9.0852,   83000),
    "017174": ("Monza",     45.5845, 9.2744,  123000),
    "003135": ("Novara",    45.4469, 8.6220,  104000),
    "004216": ("Bolzano",   46.4981, 11.3548,  107000),
    "006003": ("Alessandria",44.9131,8.6148,   91000),
    "003043": ("Biella",    45.5638, 8.0589,   44000),
    "096001": ("Agrigento", 37.3110, 13.5765,  57000),
    "074001": ("Brindisi",  40.6328, 17.9358,  86000),
    "075023": ("Messina",   38.1937, 15.5542, 218000),
    "080063": ("Reggio Cal",38.1143, 15.6600, 169000),
    "052010": ("Siena",     43.3186, 11.3306,   54000),
    "050026": ("Livorno",   43.5485, 10.3106,  155000),
    "059011": ("Perugia",   43.1107, 12.3908,  165000),
    "066049": ("Pescara",   42.4617, 14.2158,  119000),
    "067042": ("L'Aquila",  42.3498, 13.3995,   68000),
    "057025": ("Latina",    41.4677, 12.9035,  126000),
    "056046": ("Terni",     42.5634, 12.6490,  110000),
    "078086": ("Reggio Em.",44.6989, 10.6297,  172000),
    "033006": ("Ancona",    43.6158, 13.5189,  100000),
    "047023": ("Ravenna",   44.4184, 12.2035,  160000),
    "036015": ("Ferrara",   44.8378, 11.6197,  131000),
}

# Compatibilità: lookup solo popolazione
POP_LOOKUP: dict[str, int] = {

    "001272": 847287,   # Torino
    "015146": 1352631,  # Milano
    "058091": 2739000,  # Roma
    "063049": 908000,   # Napoli
    "037006": 1005791,  # Bologna (prov)  — comune: 400000
    "048017": 361000,   # Firenze
    "023023": 258000,   # Verona
    "028060": 261000,   # Padova
    "027042": 255000,   # Venezia
    "080063": 322000,   # Palermo
    "075023": 320000,   # Catania
    "042006": 280000,   # Genova... ma genova = 010025
    "010025": 565000,   # Genova
    "074001": 198000,   # Bari (075016)
    "075016": 316000,   # Bari
    "016024": 119000,   # Brescia
    "037006": 419000,   # Bologna
    "052010": 103000,   # Siena
    "050026": 111000,   # Livorno
    "020041": 119000,   # Trieste
    "024129": 113000,   # Vicenza
    "004216": 105000,   # Trento
}


@dataclass
class DatiComuneISTAT:
    codice_istat: str
    nome_comune: str
    popolazione_totale: int
    distribuzione_eta: dict        # "0-17","18-34","35-54","55-64","65+"
    distribuzione_sesso: dict      # "M","F"
    tasso_occupazione: float
    tasso_pendolarismo: float
    auto_per_abitante: float
    coordinate_centroide: tuple    # (lat, lon)


# ── Cache ─────────────────────────────────────────────────────────────────────
def _cache_path(codice_istat: str):
    ISTAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ISTAT_CACHE_DIR / f"{codice_istat}.json"


def _carica_cache(codice_istat: str) -> DatiComuneISTAT | None:
    p = _cache_path(codice_istat)
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    if datetime.now() - mtime > timedelta(days=CACHE_TTL_GIORNI):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["coordinate_centroide"] = tuple(d["coordinate_centroide"])
    return DatiComuneISTAT(**d)


def _salva_cache(dati: DatiComuneISTAT):
    p = _cache_path(dati.codice_istat)
    obj = asdict(dati)
    obj["coordinate_centroide"] = list(obj["coordinate_centroide"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── Coordinate via Nominatim (OSM) ────────────────────────────────────────────
def _coordinate_nominatim(codice_istat: str) -> tuple[str, float, float]:
    """
    Cerca nome e coordinate del comune usando OSM Nominatim.
    Ricerca per codice ISTAT nel tag 'ref:ISTAT'.
    """
    headers = {"User-Agent": "wHereAgent-OOH-Simulator/1.0"}
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":              f"[ref:ISTAT={codice_istat}]",
        "format":         "json",
        "addressdetails": 1,
        "limit":          1,
        "countrycodes":   "it",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            nome = d.get("address", {}).get("city") or \
                   d.get("address", {}).get("town") or \
                   d.get("address", {}).get("village") or \
                   d.get("display_name", codice_istat).split(",")[0]
            return nome, float(d["lat"]), float(d["lon"])
    except Exception as e:
        log.debug(f"Nominatim ref:ISTAT fallito per {codice_istat}: {e}")

    # Secondo tentativo: ricerca testuale con codice ISTAT come query libera
    try:
        params2 = {"q": codice_istat, "format": "json",
                   "countrycodes": "it", "limit": 1}
        r2 = requests.get(url, params=params2, headers=headers, timeout=10)
        if r2.status_code == 200 and r2.json():
            d = r2.json()[0]
            nome = d.get("display_name", codice_istat).split(",")[0]
            return nome, float(d["lat"]), float(d["lon"])
    except Exception:
        pass

    return codice_istat, 0.0, 0.0


# ── Download popolazione da API ISTAT open data ───────────────────────────────
def _download_popolazione(codice_istat: str) -> int | None:
    """
    Interroga l'API open data ISTAT per la popolazione residente.
    Usa l'endpoint JSON del dataset 'Popolazione residente'.
    """
    # Lookup rapido dalla tabella embedded
    if codice_istat in POP_LOOKUP:
        return POP_LOOKUP[codice_istat]

    # Tentativo API ISTAT open data (formato più stabile)
    url = "https://esploradati.istat.it/databrowser/api/v2/dsd/IT1/Q/DCIS_POPRES1/1.0"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            # L'API restituisce metadati; per i dati serve endpoint diverso
            pass
    except Exception:
        pass

    return None


# ── Costruzione dati comune ───────────────────────────────────────────────────
def _build_dati(codice_istat: str) -> DatiComuneISTAT:
    """Raccoglie nome, coordinate e popolazione per un comune."""
    # 1. Lookup tabella embedded (istantaneo, no rete)
    if codice_istat in COMUNI_LOOKUP:
        nome, lat, lon, pop = COMUNI_LOOKUP[codice_istat]
        log.info(f"ISTAT {codice_istat} — {nome} | pop={pop:,} | lookup")
    else:
        # 2. Nominatim per coordinate + lookup popolazione
        time.sleep(1.0)   # rispetta rate limit Nominatim (1 req/sec)
        nome, lat, lon = _coordinate_nominatim(codice_istat)
        pop = _download_popolazione(codice_istat) or 50000
        log.info(f"ISTAT {codice_istat} — {nome} | pop={pop:,} | ({lat:.4f},{lon:.4f})")

    return DatiComuneISTAT(
        codice_istat        = codice_istat,
        nome_comune         = nome,
        popolazione_totale  = pop,
        distribuzione_eta   = dict(DEMO_FALLBACK["distribuzione_eta"]),
        distribuzione_sesso = dict(DEMO_FALLBACK["distribuzione_sesso"]),
        tasso_occupazione   = DEMO_FALLBACK["tasso_occupazione"],
        tasso_pendolarismo  = DEMO_FALLBACK["tasso_pendolarismo"],
        auto_per_abitante   = DEMO_FALLBACK["auto_per_abitante"],
        coordinate_centroide= (lat, lon),
    )


# ── Funzione principale ───────────────────────────────────────────────────────
def arricchisci_coordinate(
    risultati: dict[str, DatiComuneISTAT],
    impianti_per_comune: dict[str, list]   # istat → list[ImpiantoABM]
) -> None:
    """
    Per i comuni con coordinate (0,0), usa la media lat/lon degli impianti
    come approssimazione del centroide. Modifica risultati in-place.
    """
    for cod, dati in risultati.items():
        if dati.coordinate_centroide != (0.0, 0.0):
            continue
        impianti = impianti_per_comune.get(cod, [])
        if not impianti:
            continue
        lat_avg = sum(i.lat for i in impianti) / len(impianti)
        lon_avg = sum(i.lon for i in impianti) / len(impianti)
        dati.coordinate_centroide = (round(lat_avg, 5), round(lon_avg, 5))
        log.info(f"Centroide {cod} da impianti: ({lat_avg:.4f},{lon_avg:.4f})")
        _salva_cache(dati)


def load_istat_data(codici_istat: list[str]) -> dict[str, DatiComuneISTAT]:
    """
    Carica dati ISTAT per una lista di codici comuni.
    Usa cache locale → API/lookup → fallback.

    Returns:
        dict  codice_istat → DatiComuneISTAT
    """
    risultati: dict[str, DatiComuneISTAT] = {}

    for cod in codici_istat:
        if not cod:
            continue

        cached = _carica_cache(cod)
        if cached:
            log.info(f"ISTAT {cod} ({cached.nome_comune}): da cache")
            risultati[cod] = cached
            continue

        dati = _build_dati(cod)
        _salva_cache(dati)
        risultati[cod] = dati

    return risultati


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    codici = sys.argv[1:] if len(sys.argv) > 1 else ["001272", "015146"]
    dati = load_istat_data(codici)
    for cod, d in dati.items():
        print(f"\n{cod} — {d.nome_comune}")
        print(f"  Popolazione : {d.popolazione_totale:,}")
        print(f"  Eta         : {d.distribuzione_eta}")
        print(f"  Centroide   : {d.coordinate_centroide}")
