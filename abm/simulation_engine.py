"""
simulation_engine.py — Coordina la simulazione di tutti gli agenti.

Strategia:
  - Parallelismo via ProcessPoolExecutor (un chunk di agenti per worker)
  - Ogni worker simula N giorni per ogni agente del suo chunk
  - I risultati parziali vengono uniti alla fine
"""
from __future__ import annotations
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .population_builder import AgentProfile
from .city_mapper import CityMap
from .campaign_loader import ImpiantoABM
from .exposure_collector import SpatialIndex, check_exposure, EsposizioneEvento
from .agent import Agent
from .config import SIM_GIORNI, SIM_WORKERS, SIM_SEED

log = logging.getLogger(__name__)


# ── Worker (deve essere top-level per ProcessPoolExecutor) ────────────────────
def _simula_chunk(args: tuple) -> list[dict]:
    """
    Simula un chunk di agenti per N giorni.
    Ritorna lista di dict (EsposizioneEvento serializzato).
    """
    profiles, city_map, impianti, n_giorni, seed_offset = args

    # Ricrea SpatialIndex nel worker (non picklabile)
    spatial_idx = SpatialIndex(impianti)

    eventi = []
    for profile in profiles:
        agente = Agent(profile, city_map, seed_offset=seed_offset)
        for giorno in range(1, n_giorni + 1):
            slots = agente.simulate_day(giorno)
            for slot in slots:
                if not slot.lat or not slot.lon or not slot.in_movimento:
                    continue
                esposizioni = check_exposure(
                    agent_id      = slot.agent_id,
                    lat           = slot.lat,
                    lon           = slot.lon,
                    mezzo         = slot.mezzo,
                    ora           = slot.ora,
                    minuto        = slot.minuto,
                    giorno        = slot.giorno,
                    spatial_index = spatial_idx,
                )
                for e in esposizioni:
                    eventi.append({
                        "agent_id":    e.agent_id,
                        "impianto_id": e.impianto_id,
                        "giorno":      e.giorno,
                        "ora":         e.ora,
                        "minuto":      e.minuto,
                        "mezzo":       e.mezzo,
                        "distanza_m":  e.distanza_m,
                    })
    return eventi


def _dict_to_evento(d: dict) -> EsposizioneEvento:
    return EsposizioneEvento(**d)


# ── Simulazione single-thread (per debug o piccoli dataset) ──────────────────
def _run_singlethread(
    agenti: list[AgentProfile],
    city_map: CityMap,
    impianti: list[ImpiantoABM],
    n_giorni: int,
) -> list[EsposizioneEvento]:
    spatial_idx = SpatialIndex(impianti)
    eventi = []
    tot = len(agenti)
    for i, profile in enumerate(agenti):
        agente = Agent(profile, city_map)
        for giorno in range(1, n_giorni + 1):
            slots = agente.simulate_day(giorno)
            for slot in slots:
                if not slot.lat or not slot.lon or not slot.in_movimento:
                    continue
                ev = check_exposure(
                    slot.agent_id, slot.lat, slot.lon,
                    slot.mezzo, slot.ora, slot.minuto, slot.giorno,
                    spatial_idx,
                )
                eventi.extend(ev)
        if (i + 1) % 100 == 0 or (i + 1) == tot:
            log.info(f"  Agenti simulati: {i+1}/{tot} | esposizioni: {len(eventi)}")
    return eventi


# ── Simulazione multi-process ─────────────────────────────────────────────────
def _run_multiprocess(
    agenti: list[AgentProfile],
    city_map: CityMap,
    impianti: list[ImpiantoABM],
    n_giorni: int,
    n_workers: int,
) -> list[EsposizioneEvento]:
    chunk_size = max(1, len(agenti) // n_workers)
    chunks = [agenti[i:i + chunk_size] for i in range(0, len(agenti), chunk_size)]

    log.info(f"Avvio {len(chunks)} worker ({n_workers} max) su {len(agenti)} agenti...")

    args_list = [
        (chunk, city_map, impianti, n_giorni, i * 1000)
        for i, chunk in enumerate(chunks)
    ]

    tutti_eventi = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_simula_chunk, args): i
                   for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            chunk_idx = futures[fut]
            try:
                partial = fut.result()
                tutti_eventi.extend(partial)
                log.info(f"  Chunk {chunk_idx+1}/{len(chunks)} completato: "
                         f"{len(partial)} esposizioni")
            except Exception as e:
                log.error(f"  Chunk {chunk_idx+1} fallito: {e}")

    return [_dict_to_evento(d) for d in tutti_eventi]


# ── Funzione principale ───────────────────────────────────────────────────────
def run_simulation(
    agenti: list[AgentProfile],
    city_map: CityMap,
    impianti: list[ImpiantoABM],
    n_giorni: int = SIM_GIORNI,
    n_workers: int = SIM_WORKERS,
    forza_singlethread: bool = False,
) -> tuple[list[EsposizioneEvento], dict]:
    """
    Esegue la simulazione completa.

    Args:
        agenti            : lista profili agenti
        city_map          : grafo OSM del comune
        impianti          : impianti OOH della campagna nel comune
        n_giorni          : giorni da simulare
        n_workers         : worker paralleli (1 = single-thread)
        forza_singlethread: ignora n_workers e usa single-thread

    Returns:
        (eventi, stats)
        - eventi : lista EsposizioneEvento
        - stats  : dict con metriche di esecuzione
    """
    t0 = time.time()
    log.info(f"Simulazione: {len(agenti)} agenti x {n_giorni} giorni "
             f"x {len(impianti)} impianti")

    if forza_singlethread or n_workers <= 1 or len(agenti) < 50:
        log.info("Modalita: single-thread")
        eventi = _run_singlethread(agenti, city_map, impianti, n_giorni)
    else:
        log.info(f"Modalita: multi-process ({n_workers} worker)")
        try:
            eventi = _run_multiprocess(agenti, city_map, impianti, n_giorni, n_workers)
        except Exception as e:
            log.warning(f"Multi-process fallito ({e}), fallback single-thread")
            eventi = _run_singlethread(agenti, city_map, impianti, n_giorni)

    durata = round(time.time() - t0, 2)
    slot_totali = len(agenti) * n_giorni * (24 * 60 // 15)

    stats = {
        "n_agenti":        len(agenti),
        "n_giorni":        n_giorni,
        "n_impianti":      len(impianti),
        "slot_totali":     slot_totali,
        "esposizioni":     len(eventi),
        "tasso_esposizione": round(len(eventi) / max(slot_totali, 1) * 100, 4),
        "durata_sec":      durata,
        "slot_per_sec":    round(slot_totali / max(durata, 0.01)),
    }

    log.info(f"Simulazione completata in {durata}s | "
             f"{stats['esposizioni']} esposizioni su {slot_totali:,} slot "
             f"({stats['tasso_esposizione']}%)")

    return eventi, stats
