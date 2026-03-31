"""
exposure_collector.py — Rileva l'esposizione agente ↔ impianto OOH.

Usa un indice spaziale R-tree per candidate selection O(log n),
poi distanza Haversine esatta per conferma.
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass

from .config import RAGGIO_PER_MEZZO, RAGGIO_DEFAULT_M, ORA_NOTTURNA_INIZIO, ORA_NOTTURNA_FINE
from .campaign_loader import ImpiantoABM

log = logging.getLogger(__name__)

# gradi ≈ metri (approssimazione per bbox R-tree)
# 1 grado lat ≈ 111km → 1m ≈ 0.000009 gradi
_M_TO_DEG = 1 / 111_000


@dataclass
class EsposizioneEvento:
    agent_id:    str
    impianto_id: str
    giorno:      int
    ora:         int
    minuto:      int
    mezzo:       str
    distanza_m:  float


# ── Haversine ─────────────────────────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distanza in metri tra due coordinate geografiche."""
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ── Indice spaziale ───────────────────────────────────────────────────────────
class SpatialIndex:
    """
    R-tree leggero basato su rtree.index.
    Fallback a ricerca lineare se rtree non disponibile.
    """
    def __init__(self, impianti: list[ImpiantoABM]):
        self._impianti = {imp.id: imp for imp in impianti}
        self._ids      = [imp.id for imp in impianti]

        try:
            from rtree import index as rtree_index
            self._idx = rtree_index.Index()
            # Raggio massimo per il bbox (auto = 80m → ~0.00072°)
            buf = RAGGIO_DEFAULT_M * 2 * _M_TO_DEG
            for i, imp in enumerate(impianti):
                self._idx.insert(i, (
                    imp.lon - buf, imp.lat - buf,
                    imp.lon + buf, imp.lat + buf,
                ))
            self._use_rtree = True
            log.debug(f"SpatialIndex R-tree: {len(impianti)} impianti")
        except ImportError:
            self._use_rtree = False
            log.warning("rtree non disponibile — uso ricerca lineare (più lento)")

    def candidati(self, lat: float, lon: float, raggio_m: float) -> list[ImpiantoABM]:
        """Ritorna impianti potenzialmente nel raggio (candidate selection)."""
        if self._use_rtree:
            buf = raggio_m * _M_TO_DEG
            hits = list(self._idx.intersection(
                (lon - buf, lat - buf, lon + buf, lat + buf)
            ))
            return [self._impianti[self._ids[i]] for i in hits]
        else:
            # Ricerca lineare: filtro grossolano per bbox prima di Haversine
            buf = raggio_m * _M_TO_DEG
            return [
                imp for imp in self._impianti.values()
                if abs(imp.lat - lat) <= buf and abs(imp.lon - lon) <= buf
            ]


# ── Controllo visibilità notturna ─────────────────────────────────────────────
def _is_notturno(ora: int) -> bool:
    return ora >= ORA_NOTTURNA_INIZIO or ora < ORA_NOTTURNA_FINE


# ── Funzione principale di check esposizione ──────────────────────────────────
def check_exposure(
    agent_id: str,
    lat: float,
    lon: float,
    mezzo: str,
    ora: int,
    minuto: int,
    giorno: int,
    spatial_index: SpatialIndex,
    config: dict = None,
) -> list[EsposizioneEvento]:
    """
    Verifica se un agente in posizione (lat, lon) è esposto a qualche impianto.

    Args:
        agent_id      : ID agente
        lat, lon      : posizione agente
        mezzo         : "piedi"|"bici"|"bus"|"auto"
        ora, minuto   : slot temporale
        giorno        : giorno della simulazione
        spatial_index : indice precostruito degli impianti
        config        : override parametri (opzionale)

    Returns:
        lista di EsposizioneEvento (vuota se nessuna esposizione)
    """
    raggio_m = RAGGIO_PER_MEZZO.get(mezzo, RAGGIO_DEFAULT_M)
    notturno = _is_notturno(ora)

    candidati = spatial_index.candidati(lat, lon, raggio_m)
    eventi: list[EsposizioneEvento] = []

    for imp in candidati:
        # Impianti non illuminati non visibili di notte
        if notturno and not imp.illuminato:
            continue

        # Raggio specifico per questo impianto + mezzo
        raggio_imp = imp.raggio_per(mezzo)

        dist = haversine_m(lat, lon, imp.lat, imp.lon)
        if dist <= raggio_imp:
            eventi.append(EsposizioneEvento(
                agent_id    = agent_id,
                impianto_id = imp.id,
                giorno      = giorno,
                ora         = ora,
                minuto      = minuto,
                mezzo       = mezzo,
                distanza_m  = round(dist, 1),
            ))

    return eventi


# ── Aggregazione esposizioni ───────────────────────────────────────────────────
def aggrega_per_impianto(
    eventi: list[EsposizioneEvento],
    n_agenti_totale: int,
    popolazione: int,
) -> dict:
    """
    Calcola OTS, Reach univoco, Frequenza, GRP, Copertura per ogni impianto.

    Returns:
        dict  impianto_id → metriche
    """
    from collections import defaultdict

    # Raggruppa per impianto
    per_imp: dict[str, list[EsposizioneEvento]] = defaultdict(list)
    for e in eventi:
        per_imp[e.impianto_id].append(e)

    risultati = {}
    for imp_id, evs in per_imp.items():
        ots        = len(evs)
        agenti_unici = len({e.agent_id for e in evs})

        # Scala reach alla popolazione reale
        scala      = popolazione / max(n_agenti_totale, 1)
        reach_reale= round(agenti_unici * scala)
        freq       = round(ots / max(agenti_unici, 1), 2)
        cop_pct    = round(reach_reale / max(popolazione, 1) * 100, 2)
        grp        = round(cop_pct * freq, 1)

        risultati[imp_id] = {
            "ots_totali":    ots,
            "reach_univoco": reach_reale,
            "frequency_media": freq,
            "grp":           grp,
            "copertura_pct": cop_pct,
        }

    return risultati


def aggrega_per_ora(eventi: list[EsposizioneEvento]) -> dict:
    """Raggruppa esposizioni per impianto e ora del giorno."""
    from collections import defaultdict
    per_imp_ora: dict[tuple, set] = defaultdict(set)
    per_imp_ora_count: dict[tuple, int] = defaultdict(int)

    for e in eventi:
        key = (e.impianto_id, e.ora)
        per_imp_ora[key].add(e.agent_id)
        per_imp_ora_count[key] += 1

    risultati = []
    for (imp_id, ora), agenti in per_imp_ora.items():
        risultati.append({
            "impianto_id":  imp_id,
            "ora":          ora,
            "n_esposizioni": per_imp_ora_count[(imp_id, ora)],
            "n_agenti_unici": len(agenti),
        })
    return risultati


def aggrega_per_mezzo(eventi: list[EsposizioneEvento]) -> dict:
    """Raggruppa esposizioni per impianto e mezzo di trasporto."""
    from collections import defaultdict
    per_imp_mezzo: dict[tuple, set] = defaultdict(set)
    per_imp_mezzo_count: dict[tuple, int] = defaultdict(int)

    for e in eventi:
        key = (e.impianto_id, e.mezzo)
        per_imp_mezzo[key].add(e.agent_id)
        per_imp_mezzo_count[key] += 1

    risultati = []
    for (imp_id, mezzo), agenti in per_imp_mezzo.items():
        risultati.append({
            "impianto_id":   imp_id,
            "mezzo":         mezzo,
            "n_esposizioni": per_imp_mezzo_count[(imp_id, mezzo)],
            "n_agenti_unici": len(agenti),
        })
    return risultati


def aggrega_frequenza_individuale(eventi: list[EsposizioneEvento]) -> list[dict]:
    """Frequenza reale per ogni coppia agente-impianto."""
    from collections import defaultdict
    contatore: dict[tuple, list[EsposizioneEvento]] = defaultdict(list)
    for e in eventi:
        contatore[(e.agent_id, e.impianto_id)].append(e)

    risultati = []
    for (agent_id, imp_id), evs in contatore.items():
        risultati.append({
            "agent_id":    agent_id,
            "impianto_id": imp_id,
            "n_esposizioni": len(evs),
        })
    return risultati


if __name__ == "__main__":
    """Test rapido con impianti e agente fittizi."""
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    from .campaign_loader import ImpiantoABM
    from .config import RAGGIO_PER_MEZZO

    # Impianti fittizi intorno a Piazza Castello, Torino
    imp_test = [
        ImpiantoABM("BB-T01", 1, "billboard", 45.0703, 7.6869,
                    "001272", 1, "6x3", "Piazza Castello", True, False,
                    dict(RAGGIO_PER_MEZZO)),
        ImpiantoABM("BB-T02", 1, "billboard", 45.0750, 7.6900,
                    "001272", 1, "6x3", "Via Po", True, False,
                    dict(RAGGIO_PER_MEZZO)),
    ]

    idx = SpatialIndex(imp_test)

    # Agente in Piazza Castello a piedi
    eventi = check_exposure("AG001", 45.0705, 7.6872, "piedi", 9, 15, 1, idx)
    print(f"Esposizioni rilevate: {len(eventi)}")
    for e in eventi:
        print(f"  {e.impianto_id} dist={e.distanza_m}m mezzo={e.mezzo}")
