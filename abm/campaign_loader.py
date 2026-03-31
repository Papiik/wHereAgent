"""
campaign_loader.py — Carica impianti della campagna da MySQL.
Converte le coordinate italiane (virgola) in float standard.
"""
from __future__ import annotations
from dataclasses import dataclass
import mysql.connector
from .config import DB_CONFIG, TIPO_MAP, RAGGIO_DEFAULT_M, RAGGIO_PER_MEZZO


@dataclass
class ImpiantoABM:
    id: str
    id_campagna: int
    tipo: str                   # normalizzato via TIPO_MAP
    lat: float
    lon: float
    istat_comune: str
    id_provincia: int
    formato: str
    indirizzo: str
    illuminato: bool            # default True (non in tabella, assumiamo illuminati)
    digitale: bool
    raggio_visibilita_m: dict   # raggio per mezzo di trasporto

    def raggio_per(self, mezzo: str) -> int:
        return self.raggio_visibilita_m.get(mezzo, RAGGIO_DEFAULT_M)


def _normalizza_tipo(raw: str) -> str:
    if not raw:
        return "billboard"
    raw_low = raw.strip().lower()
    for key, val in TIPO_MAP.items():
        if key in raw_low:
            return val
    return "billboard"


def _parse_coord(val: str) -> float:
    """Converte coordinate italiane con virgola in float. Ritorna 0.0 se malformata."""
    if not val:
        return 0.0
    try:
        return float(str(val).strip().replace(",", "."))
    except ValueError:
        return 0.0


def _is_digitale(tipo_det: str, circuito: str) -> bool:
    if not tipo_det and not circuito:
        return False
    testo = f"{tipo_det or ''} {circuito or ''}".lower()
    return any(k in testo for k in ["dooh", "digital", "led", "digitale"])


def load_campaign(id_campagna: int, conn_params: dict = None) -> dict:
    """
    Carica tutti gli impianti di una campagna da MySQL.

    Returns:
        {
            "impianti":      list[ImpiantoABM],
            "comuni_istat":  list[str],          # codici ISTAT univoci
            "id_campagna":   int,
        }
    """
    params = conn_params or DB_CONFIG
    conn = mysql.connector.connect(**params)
    cur  = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            id, idCampagna, CodiceInpe, Cimasa,
            istatComune, idProvincia,
            TipoImpiantoDet, TipoImpiantoDesc,
            circuito, Formato, Indirizzo,
            Latitudine, Longitudine
        FROM campaigns
        WHERE idCampagna = %s
          AND Latitudine IS NOT NULL
          AND Longitudine IS NOT NULL
          AND Latitudine  != ''
          AND Longitudine != ''
    """, (id_campagna,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    impianti: list[ImpiantoABM] = []
    for r in rows:
        lat = _parse_coord(r["Latitudine"])
        lon = _parse_coord(r["Longitudine"])
        if lat == 0.0 or lon == 0.0:
            continue

        tipo     = _normalizza_tipo(r.get("TipoImpiantoDet") or r.get("TipoImpiantoDesc"))
        digitale = _is_digitale(r.get("TipoImpiantoDet"), r.get("circuito"))

        # Raggio visibilità per mezzo (billboard ha raggio maggiore)
        raggio = dict(RAGGIO_PER_MEZZO)
        if tipo == "billboard":
            raggio = {k: int(v * 1.5) for k, v in raggio.items()}

        impianti.append(ImpiantoABM(
            id               = str(r["id"]),
            id_campagna      = id_campagna,
            tipo             = tipo,
            lat              = lat,
            lon              = lon,
            istat_comune     = str(r["istatComune"] or ""),
            id_provincia     = int(r["idProvincia"] or 0),
            formato          = str(r["Formato"] or ""),
            indirizzo        = str(r["Indirizzo"] or ""),
            illuminato       = True,   # default: tutti illuminati
            digitale         = digitale,
            raggio_visibilita_m = raggio,
        ))

    comuni = list({imp.istat_comune for imp in impianti if imp.istat_comune})

    return {
        "impianti":     impianti,
        "comuni_istat": comuni,
        "id_campagna":  id_campagna,
    }


if __name__ == "__main__":
    import sys
    id_c = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    risultato = load_campaign(id_c)
    print(f"Campagna {id_c}: {len(risultato['impianti'])} impianti")
    print(f"Comuni ISTAT: {risultato['comuni_istat']}")
    for imp in risultato["impianti"][:5]:
        print(f"  {imp.id} | {imp.tipo:10} | {imp.lat},{imp.lon} | {imp.istat_comune}")
