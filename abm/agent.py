"""
agent.py — Classe Agent: simula la giornata di un singolo agente.

L'agente si muove sulla rete OSM seguendo la sua routine:
  mattina  : casa → destinazione principale (lavoro/scuola/ecc.)
  pranzo   : eventuale uscita dalla destinazione principale
  pomeriggio: ritorno a casa
  sera     : eventuale attività serale
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .population_builder import AgentProfile
from .city_mapper import CityMap
from .config import SLOT_MINUTI, SLOT_PER_GIORNO

log = logging.getLogger(__name__)


@dataclass
class MovimentoSlot:
    """Posizione dell'agente in un intervallo di SLOT_MINUTI minuti."""
    agent_id:   str
    giorno:     int
    ora:        int
    minuto:     int       # 0, 15, 30, 45
    lat:        float
    lon:        float
    nodo_osm:   int
    mezzo:      str
    in_movimento: bool


class Agent:
    def __init__(
        self,
        profile: AgentProfile,
        city_map: CityMap,
        seed_offset: int = 0,
    ):
        self.p    = profile
        self.cm   = city_map
        self._rng = np.random.default_rng(abs(hash(profile.agent_id)) + seed_offset)

        # Nodo OSM della casa (già assegnato nel profilo)
        self._nodo_casa = profile.nodo_casa

        # Sceglie destinazione principale una volta sola (stabile tra i giorni)
        network = self._network_per_mezzo(profile.mezzo_preferito)
        self._nodo_dest = self._scegli_destinazione(network)

        # Cache percorsi per evitare ricalcoli
        self._route_cache: dict[tuple, list[int]] = {}

    # ── Rete OSM da usare per mezzo ───────────────────────────────────────────
    @staticmethod
    def _network_per_mezzo(mezzo: str) -> str:
        return "drive" if mezzo == "auto" else "walk"

    # ── Sceglie nodo destinazione casuale nella rete ──────────────────────────
    def _scegli_destinazione(self, network: str) -> int:
        g = self.cm.grafo_walk if network == "walk" else self.cm.grafo_drive
        nodi = list(g.nodes())
        return int(self._rng.choice(nodi))

    # ── Percorso con cache ────────────────────────────────────────────────────
    def _percorso(self, orig: int, dest: int, network: str) -> list[int]:
        key = (orig, dest, network)
        if key not in self._route_cache:
            self._route_cache[key] = self.cm.get_route(orig, dest, network)
        return self._route_cache[key]

    # ── Coordinate di un nodo OSM ─────────────────────────────────────────────
    def _coord_nodo(self, nodo: int, network: str) -> tuple[float, float]:
        g = self.cm.grafo_walk if network == "walk" else self.cm.grafo_drive
        if nodo in g.nodes:
            d = g.nodes[nodo]
            return float(d["y"]), float(d["x"])
        return self.cm.coordinate_centroide if hasattr(self.cm, "coordinate_centroide") \
               else (0.0, 0.0)

    # ── Lunghezza percorso in minuti ──────────────────────────────────────────
    def _durata_minuti(self, route: list[int], network: str) -> int:
        """Stima tempo di percorrenza in minuti basato su lunghezza OSM."""
        g = self.cm.grafo_walk if network == "walk" else self.cm.grafo_drive
        velocita_kmh = {"piedi": 5, "bici": 14, "bus": 20, "auto": 30}
        v = velocita_kmh.get(self.p.mezzo_preferito, 15)
        lunghezza_m = 0.0
        for i in range(len(route) - 1):
            u, v_nodo = route[i], route[i + 1]
            if g.has_edge(u, v_nodo):
                dati_edge = g[u][v_nodo]
                # MultiDiGraph: prendi il primo arco
                arco = list(dati_edge.values())[0]
                lunghezza_m += float(arco.get("length", 100))
        durata = (lunghezza_m / 1000) / v * 60
        return max(int(durata), SLOT_MINUTI)

    # ── Posizione interpolata lungo un percorso ───────────────────────────────
    def _posizione_in_percorso(
        self, route: list[int], network: str,
        slot_corrente: int, slot_inizio: int, slot_fine: int
    ) -> tuple[float, float, int]:
        """
        Ritorna (lat, lon, nodo) interpolando la posizione dell'agente
        lungo il percorso nel slot temporale corrente.
        """
        if not route or slot_fine <= slot_inizio:
            c = self._coord_nodo(route[-1] if route else self._nodo_casa, network)
            return c[0], c[1], route[-1] if route else self._nodo_casa

        frac = (slot_corrente - slot_inizio) / (slot_fine - slot_inizio)
        frac = min(max(frac, 0.0), 1.0)
        idx  = min(int(frac * (len(route) - 1)), len(route) - 1)
        nodo = route[idx]
        lat, lon = self._coord_nodo(nodo, network)
        return lat, lon, nodo

    # ── Simulazione giornata ───────────────────────────────────────────────────
    def simulate_day(self, giorno: int) -> list[MovimentoSlot]:
        """
        Simula una giornata completa dell'agente.
        Ritorna lista di MovimentoSlot ordinata per slot temporale.
        """
        network = self._network_per_mezzo(self.p.mezzo_preferito)
        mezzo   = self.p.mezzo_preferito

        # Percorsi principali
        route_andata  = self._percorso(self._nodo_casa, self._nodo_dest, network)
        route_ritorno = self._percorso(self._nodo_dest, self._nodo_casa, network)

        dur_andata  = self._durata_minuti(route_andata, network)
        dur_ritorno = self._durata_minuti(route_ritorno, network)

        # Slot di partenza e arrivo (in slot da 15 min)
        slot_partenza = self.p.ora_partenza * (60 // SLOT_MINUTI)
        slot_arrivo   = slot_partenza + math.ceil(dur_andata / SLOT_MINUTI)
        slot_rientro  = self.p.ora_rientro * (60 // SLOT_MINUTI)
        slot_arrivo_casa = slot_rientro + math.ceil(dur_ritorno / SLOT_MINUTI)

        # Attività serale
        slot_serale_inizio = None
        slot_serale_fine   = None
        nodo_serale        = None
        route_serale_and   = []
        route_serale_rit   = []
        if self.p.ha_attivita_serale:
            ora_uscita = min(self.p.ora_rientro + 1 + int(self._rng.integers(0, 2)), 22)
            slot_serale_inizio = ora_uscita * (60 // SLOT_MINUTI)
            nodo_serale = self._scegli_destinazione(network)
            route_serale_and = self._percorso(self._nodo_casa, nodo_serale, network)
            route_serale_rit = self._percorso(nodo_serale, self._nodo_casa, network)
            dur_ser = self._durata_minuti(route_serale_and, network)
            slot_serale_arrivo = slot_serale_inizio + math.ceil(dur_ser / SLOT_MINUTI)
            # Permanenza serale: 45-120 min
            perm_ser = int(self._rng.integers(3, 9))
            slot_serale_fine = slot_serale_arrivo + perm_ser

        slots: list[MovimentoSlot] = []

        for slot in range(SLOT_PER_GIORNO):
            ora    = (slot * SLOT_MINUTI) // 60
            minuto = (slot * SLOT_MINUTI) % 60

            # ── Prima della partenza: a casa ─────────────────────────────────
            if slot < slot_partenza:
                lat, lon = self._coord_nodo(self._nodo_casa, network)
                nodo = self._nodo_casa
                in_mov = False
                m = mezzo

            # ── In viaggio verso destinazione ────────────────────────────────
            elif slot_partenza <= slot < slot_arrivo:
                lat, lon, nodo = self._posizione_in_percorso(
                    route_andata, network, slot, slot_partenza, slot_arrivo
                )
                in_mov = True
                m = mezzo

            # ── Alla destinazione principale ──────────────────────────────────
            elif slot_arrivo <= slot < slot_rientro:
                lat, lon = self._coord_nodo(self._nodo_dest, network)
                nodo = self._nodo_dest
                in_mov = False
                m = "piedi"   # a piedi mentre è sul posto

            # ── In viaggio di ritorno ─────────────────────────────────────────
            elif slot_rientro <= slot < slot_arrivo_casa:
                lat, lon, nodo = self._posizione_in_percorso(
                    route_ritorno, network, slot, slot_rientro, slot_arrivo_casa
                )
                in_mov = True
                m = mezzo

            # ── A casa nel pomeriggio/sera ────────────────────────────────────
            elif slot_arrivo_casa <= slot:
                # Controlla attività serale
                if (slot_serale_inizio and nodo_serale and
                        slot_serale_inizio <= slot < slot_serale_fine):
                    if slot < slot_serale_inizio + len(route_serale_and):
                        lat, lon, nodo = self._posizione_in_percorso(
                            route_serale_and, network, slot,
                            slot_serale_inizio, slot_serale_inizio + len(route_serale_and)
                        )
                        in_mov = True
                    elif slot_serale_fine and slot >= slot_serale_fine - len(route_serale_rit):
                        lat, lon, nodo = self._posizione_in_percorso(
                            route_serale_rit, network, slot,
                            slot_serale_fine - len(route_serale_rit), slot_serale_fine
                        )
                        in_mov = True
                    else:
                        lat, lon = self._coord_nodo(nodo_serale, network)
                        nodo = nodo_serale
                        in_mov = False
                    m = mezzo
                else:
                    lat, lon = self._coord_nodo(self._nodo_casa, network)
                    nodo = self._nodo_casa
                    in_mov = False
                    m = "piedi"
            else:
                lat, lon = self._coord_nodo(self._nodo_casa, network)
                nodo = self._nodo_casa
                in_mov = False
                m = mezzo

            slots.append(MovimentoSlot(
                agent_id    = self.p.agent_id,
                giorno      = giorno,
                ora         = ora,
                minuto      = minuto,
                lat         = round(lat, 6),
                lon         = round(lon, 6),
                nodo_osm    = nodo,
                mezzo       = m,
                in_movimento= in_mov,
            ))

        return slots
