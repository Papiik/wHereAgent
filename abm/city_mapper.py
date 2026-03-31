"""
city_mapper.py — Carica il grafo stradale OSM per un comune.

Usa osmnx per scaricare strade e rete pedonale.
Cache su disco in formato GraphML per evitare re-download.
"""
from __future__ import annotations
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import osmnx as ox
import numpy as np

from .config import OSM_CACHE_DIR

log = logging.getLogger(__name__)

# osmnx: disabilita log verboso
ox.settings.log_console = False
ox.settings.use_cache   = True


@dataclass
class CityMap:
    nome: str
    codice_istat: str
    grafo_drive: nx.MultiDiGraph   # rete auto/bus
    grafo_walk:  nx.MultiDiGraph   # rete pedonale/bici
    bbox: tuple                    # (minlat, minlon, maxlat, maxlon)

    # ── Nodo OSM più vicino ───────────────────────────────────────────────────
    def nearest_node(self, lat: float, lon: float, network: str = "walk") -> int:
        g = self.grafo_walk if network == "walk" else self.grafo_drive
        try:
            return ox.nearest_nodes(g, lon, lat)   # osmnx vuole (lon, lat)
        except ImportError:
            # Fallback senza scikit-learn: ricerca lineare sui nodi
            best_node, best_dist = None, float("inf")
            for node, data in g.nodes(data=True):
                d = (data["y"] - lat)**2 + (data["x"] - lon)**2
                if d < best_dist:
                    best_dist, best_node = d, node
            return best_node

    # ── Shortest path tra due nodi ────────────────────────────────────────────
    def get_route(self, orig: int, dest: int, network: str = "walk") -> list[int]:
        g = self.grafo_walk if network == "walk" else self.grafo_drive
        try:
            return nx.shortest_path(g, orig, dest, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [orig, dest]

    # ── Coordinate di una lista di nodi ──────────────────────────────────────
    def route_coords(self, node_list: list[int], network: str = "walk") -> list[tuple]:
        g = self.grafo_walk if network == "walk" else self.grafo_drive
        coords = []
        for n in node_list:
            if n in g.nodes:
                d = g.nodes[n]
                coords.append((d["y"], d["x"]))   # (lat, lon)
        return coords

    # ── Lista nodi casuali (per destinazioni) ────────────────────────────────
    def random_nodes(self, n: int, network: str = "walk", seed: int = None) -> list[int]:
        g = self.grafo_walk if network == "walk" else self.grafo_drive
        rng = np.random.default_rng(seed)
        nodi = list(g.nodes())
        idx  = rng.choice(len(nodi), size=min(n, len(nodi)), replace=False)
        return [nodi[i] for i in idx]

    # ── Verifica se le coordinate sono dentro la bbox ────────────────────────
    def contains(self, lat: float, lon: float) -> bool:
        minlat, minlon, maxlat, maxlon = self.bbox
        return minlat <= lat <= maxlat and minlon <= lon <= maxlon


# ── Cache ─────────────────────────────────────────────────────────────────────
def _cache_path(codice_istat: str) -> Path:
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return OSM_CACHE_DIR / f"{codice_istat}.pkl"


def _salva_cache(city_map: CityMap):
    p = _cache_path(city_map.codice_istat)
    with open(p, "wb") as f:
        pickle.dump(city_map, f)
    log.info(f"OSM cache salvata: {p.name}")


def _carica_cache(codice_istat: str) -> CityMap | None:
    p = _cache_path(codice_istat)
    if not p.exists():
        return None
    log.info(f"OSM {codice_istat}: carico da cache...")
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Download OSM ──────────────────────────────────────────────────────────────
def _download_city(nome: str, lat: float, lon: float,
                   codice_istat: str, dist_m: int = 8000) -> CityMap:
    """
    Scarica il grafo OSM entro dist_m metri dal centroide del comune.
    dist_m default 8000m copre la maggior parte dei comuni medi italiani.
    """
    log.info(f"Download OSM per {nome} ({codice_istat}), raggio={dist_m}m...")

    # Rete pedonale (include bici e piedi)
    g_walk = ox.graph_from_point(
        (lat, lon),
        dist=dist_m,
        network_type="walk",
        simplify=True,
    )

    # Rete veicolare (auto e bus)
    g_drive = ox.graph_from_point(
        (lat, lon),
        dist=dist_m,
        network_type="drive",
        simplify=True,
    )

    # Calcola bbox dal grafo walk (più esteso)
    nodes_walk = ox.graph_to_gdfs(g_walk, edges=False)
    bbox = (
        float(nodes_walk.geometry.y.min()),
        float(nodes_walk.geometry.x.min()),
        float(nodes_walk.geometry.y.max()),
        float(nodes_walk.geometry.x.max()),
    )

    log.info(f"  Nodi walk={g_walk.number_of_nodes()}, drive={g_drive.number_of_nodes()}")

    return CityMap(
        nome         = nome,
        codice_istat = codice_istat,
        grafo_drive  = g_drive,
        grafo_walk   = g_walk,
        bbox         = bbox,
    )


# ── Funzione principale ───────────────────────────────────────────────────────
def load_city_map(
    codice_istat: str,
    nome: str,
    lat: float,
    lon: float,
    dist_m: int = 8000,
    forza_download: bool = False,
) -> CityMap:
    """
    Carica il grafo OSM di un comune.
    Usa cache pickle su disco; scarica se non presente.

    Args:
        codice_istat  : codice ISTAT del comune
        nome          : nome del comune (per log)
        lat, lon      : coordinate centroide
        dist_m        : raggio in metri dal centroide (default 8km)
        forza_download: ignora cache e riscarica
    """
    if not forza_download:
        cached = _carica_cache(codice_istat)
        if cached:
            log.info(f"OSM {nome}: da cache (nodi={cached.grafo_walk.number_of_nodes()})")
            return cached

    if lat == 0.0 or lon == 0.0:
        raise ValueError(f"Coordinate mancanti per {codice_istat} ({nome}). "
                         f"Popola COMUNI_LOOKUP in istat_loader.py o fornisci lat/lon.")

    city_map = _download_city(nome, lat, lon, codice_istat, dist_m)
    _salva_cache(city_map)
    return city_map


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Test: carica Torino
    cm = load_city_map("001272", "Torino", 45.0703, 7.6869, dist_m=5000)
    print(f"\nCityMap: {cm.nome}")
    print(f"  Nodi walk : {cm.grafo_walk.number_of_nodes()}")
    print(f"  Nodi drive: {cm.grafo_drive.number_of_nodes()}")
    print(f"  BBox      : {cm.bbox}")

    # Test nearest_node
    nodo = cm.nearest_node(45.0703, 7.6869)
    print(f"  Nodo più vicino al centroide: {nodo}")

    # Test route
    nodi_rnd = cm.random_nodes(2, seed=42)
    route = cm.get_route(nodi_rnd[0], nodi_rnd[1])
    print(f"  Percorso test: {len(route)} nodi")
