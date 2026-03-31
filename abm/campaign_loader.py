"""
campaign_loader.py — Carica impianti della campagna da MySQL (tabella campagneresult).
"""
from __future__ import annotations
from dataclasses import dataclass
import mysql.connector
from .config import DB_CONFIG, TIPO_MAP, RAGGIO_DEFAULT_M, RAGGIO_PER_MEZZO


@dataclass
class ImpiantoABM:
    id: str
    id_campagna: int
    tipo: str
    lat: float
    lon: float
    istat_comune: str
    id_provincia: int
    formato: str
    indirizzo: str
    illuminato: bool
    digitale: bool
    raggio_visibilita_m: dict

    def raggio_per(self, mezzo: str) -> int:
        return self.raggio_visibilita_m.get(mezzo, RAGGIO_DEFAULT_M)


@dataclass
class ComuneInfo:
    """Dati geografici/demografici del comune estratti da campagneresult."""
    istat_comune:    str
    nome_comune:     str
    provincia:       str
    regione:         str
    ripartizione_geo: str
    abitanti:        int


def _normalizza_tipo(raw: str) -> str:
    if not raw:
        return "billboard"
    raw_low = raw.strip().lower()
    for key, val in TIPO_MAP.items():
        if key in raw_low:
            return val
    return "billboard"


def _parse_coord(val) -> float:
    """Converte coordinata (stringa con punto o virgola, o numero) in float."""
    if val is None:
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
    Carica impianti e dati comuni da campagneresult.

    Returns:
        {
            "impianti":      list[ImpiantoABM],
            "comuni_istat":  list[str],
            "comuni_info":   dict[str, ComuneInfo],   # istat → dati comune
            "id_campagna":   int,
        }
    """
    params = conn_params or DB_CONFIG
    conn = mysql.connector.connect(**params)
    cur  = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT
            id, idCampagna,
            istatComune, idProvincia,
            TipoImpiantoDet, TipoImpiantoDesc,
            circuito, Formato, Indirizzo,
            Latitudine, Longitudine,
            comune, provincia, regione, ripartizionegeo,
            abitanti
        FROM campagneresult
        WHERE idCampagna = %s
          AND Latitudine  IS NOT NULL
          AND Longitudine IS NOT NULL
          AND Latitudine  != 0
          AND Longitudine != 0
    """, (id_campagna,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    impianti: list[ImpiantoABM] = []
    comuni_info: dict[str, ComuneInfo] = {}

    for r in rows:
        lat = _parse_coord(r["Latitudine"])
        lon = _parse_coord(r["Longitudine"])
        if lat == 0.0 or lon == 0.0:
            continue

        tipo     = _normalizza_tipo(r.get("TipoImpiantoDet") or r.get("TipoImpiantoDesc"))
        digitale = _is_digitale(r.get("TipoImpiantoDet"), r.get("circuito"))
        istat    = str(r["istatComune"] or "")

        raggio = dict(RAGGIO_PER_MEZZO)
        if tipo == "billboard":
            raggio = {k: int(v * 1.5) for k, v in raggio.items()}

        impianti.append(ImpiantoABM(
            id                  = str(r["id"]),
            id_campagna         = id_campagna,
            tipo                = tipo,
            lat                 = lat,
            lon                 = lon,
            istat_comune        = istat,
            id_provincia        = int(r["idProvincia"] or 0),
            formato             = str(r["Formato"] or ""),
            indirizzo           = str(r["Indirizzo"] or ""),
            illuminato          = digitale,
            digitale            = digitale,
            raggio_visibilita_m = raggio,
        ))

        # Salva info comune (una volta per codice ISTAT)
        if istat and istat not in comuni_info:
            comuni_info[istat] = ComuneInfo(
                istat_comune     = istat,
                nome_comune      = str(r["comune"] or istat),
                provincia        = str(r["provincia"] or ""),
                regione          = str(r["regione"] or ""),
                ripartizione_geo = str(r["ripartizionegeo"] or ""),
                abitanti         = int(r["abitanti"] or 0),
            )

    comuni = list({imp.istat_comune for imp in impianti if imp.istat_comune})

    return {
        "impianti":     impianti,
        "comuni_istat": comuni,
        "comuni_info":  comuni_info,
        "id_campagna":  id_campagna,
    }


if __name__ == "__main__":
    import sys
    id_c = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    risultato = load_campaign(id_c)
    print(f"Campagna {id_c}: {len(risultato['impianti'])} impianti, "
          f"{len(risultato['comuni_istat'])} comuni")
    for istat, ci in list(risultato["comuni_info"].items())[:5]:
        print(f"  {istat} | {ci.nome_comune} | {ci.provincia} | {ci.regione} | ab={ci.abitanti:,}")
