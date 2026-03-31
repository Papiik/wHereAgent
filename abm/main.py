"""
main.py — Entry point del sistema ABM OOH.

Uso:
    python -m abm.main --campagna 569
    python -m abm.main --campagna 569 --comuni 017174 001272
    python -m abm.main --campagna 569 --scale 0.01 --giorni 1 --mode 1
    python -m abm.main --campagna 569 --singlethread
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from collections import defaultdict

from .config import (
    DB_CONFIG, SIM_SCALE, SIM_GIORNI, SIM_WORKERS, SIM_SEED, PROFILO_MODE
)
from .campaign_loader import load_campaign
from .istat_loader import load_istat_data, arricchisci_coordinate
from .city_mapper import load_city_map, calc_dist_m
from .population_builder import build_population
from .simulation_engine import run_simulation
from .db_writer import (
    crea_tabelle, apri_run, chiudi_run, segna_errore,
    salva_risultati, stampa_riepilogo,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ABM OOH Simulator")
    parser.add_argument("--campagna",    type=int, required=True,
                        help="ID campagna MySQL")
    parser.add_argument("--comuni",      nargs="*",
                        help="Filtra su specifici codici ISTAT (es. 017174 001272)")
    parser.add_argument("--scale",       type=float, default=SIM_SCALE,
                        help="Frazione popolazione da simulare (0.01-0.10)")
    parser.add_argument("--giorni",      type=int, default=SIM_GIORNI,
                        help="Giorni da simulare")
    parser.add_argument("--workers",     type=int, default=SIM_WORKERS,
                        help="Worker paralleli")
    parser.add_argument("--mode",        type=int, default=PROFILO_MODE,
                        choices=[0, 1, 2],
                        help="0=Claude 1=Python 2=Entrambi+analisi")
    parser.add_argument("--singlethread", action="store_true",
                        help="Forza single-thread (utile per debug)")
    parser.add_argument("--no-db",       action="store_true",
                        help="Non scrive su MySQL (solo output console)")
    args = parser.parse_args()

    t_totale = time.time()
    log.info(f"=== ABM OOH Simulator | campagna={args.campagna} "
             f"scale={args.scale} giorni={args.giorni} mode={args.mode} ===")

    # ── 1. Tabelle DB ─────────────────────────────────────────────────────────
    if not args.no_db:
        crea_tabelle()

    # ── 2. Carica campagna ────────────────────────────────────────────────────
    log.info("Caricamento campagna...")
    camp = load_campaign(args.campagna)
    impianti_tutti  = camp["impianti"]
    comuni_campagna = camp["comuni_istat"]
    comuni_info_db  = camp["comuni_info"]

    if args.comuni:
        comuni_da_simulare = [c for c in comuni_campagna if c in args.comuni]
    else:
        comuni_da_simulare = comuni_campagna

    log.info(f"Comuni da simulare: {len(comuni_da_simulare)} / {len(comuni_campagna)}")

    # ── 3. Dati ISTAT ─────────────────────────────────────────────────────────
    log.info("Caricamento dati ISTAT...")
    imp_per_comune = defaultdict(list)
    for imp in impianti_tutti:
        imp_per_comune[imp.istat_comune].append(imp)

    dati_istat = load_istat_data(comuni_da_simulare, imp_per_comune, comuni_info_db)
    arricchisci_coordinate(dati_istat, imp_per_comune)

    # ── 4. Loop per comune ────────────────────────────────────────────────────
    for codice_istat in comuni_da_simulare:
        dati = dati_istat.get(codice_istat)
        if not dati:
            log.warning(f"Dati ISTAT mancanti per {codice_istat}, skip")
            continue

        impianti_comune = imp_per_comune.get(codice_istat, [])
        if not impianti_comune:
            log.warning(f"Nessun impianto per {codice_istat}, skip")
            continue

        # Usa centroide degli impianti reali (più affidabile del centroide ISTAT)
        lat_imp = sum(i.lat for i in impianti_comune) / len(impianti_comune)
        lon_imp = sum(i.lon for i in impianti_comune) / len(impianti_comune)

        lat_istat, lon_istat = dati.coordinate_centroide
        # Se centroide ISTAT è molto lontano dagli impianti, usa quello degli impianti
        if abs(lat_imp - lat_istat) > 0.3 or abs(lon_imp - lon_istat) > 0.3:
            log.warning(f"Centroide ISTAT lontano dagli impianti per {dati.nome_comune} "
                        f"— uso centroide impianti ({lat_imp:.4f},{lon_imp:.4f})")
            lat, lon = lat_imp, lon_imp
        else:
            lat, lon = lat_istat, lon_istat

        if lat == 0.0 or lon == 0.0:
            log.warning(f"Coordinate mancanti per {dati.nome_comune}, skip")
            continue

        log.info(f"\n--- Comune: {dati.nome_comune} ({codice_istat}) | "
                 f"pop={dati.popolazione_totale:,} | "
                 f"impianti={len(impianti_comune)} ---")

        run_id = None
        try:
            # 4a. Mappa OSM
            dist_m   = calc_dist_m(lat, lon, impianti_comune)
            city_map = load_city_map(
                codice_istat, dati.nome_comune, lat, lon, dist_m=dist_m
            )

            # 4b. Popolazione agenti
            import os
            os.environ["SIM_SCALE"] = str(args.scale)
            from . import config as cfg
            cfg.SIM_SCALE = args.scale

            agenti, analisi = build_population(dati, city_map, mode=args.mode)
            log.info(f"Agenti generati: {len(agenti)}")

            # 4c. Apri run su DB
            config_snapshot = {
                "scale": args.scale, "giorni": args.giorni,
                "workers": args.workers, "mode": args.mode,
                "seed": SIM_SEED,
            }
            if not args.no_db:
                run_id = apri_run(
                    args.campagna, codice_istat, dati.nome_comune,
                    len(agenti), args.giorni, len(impianti_comune),
                    args.mode, config_snapshot,
                )

            # 4d. Simulazione
            eventi, stats = run_simulation(
                agenti        = agenti,
                city_map      = city_map,
                impianti      = impianti_comune,
                n_giorni      = args.giorni,
                n_workers     = args.workers,
                forza_singlethread = args.singlethread,
            )

            # 4e. Salva risultati
            metriche = None
            if not args.no_db and run_id:
                metriche = salva_risultati(
                    run_id      = run_id,
                    id_campagna = args.campagna,
                    eventi      = eventi,
                    agenti      = agenti,
                    impianti    = impianti_comune,
                    popolazione = dati.popolazione_totale,
                )
                chiudi_run(run_id, stats["durata_sec"], stats["esposizioni"])
            else:
                from .exposure_collector import aggrega_per_impianto
                metriche = aggrega_per_impianto(
                    eventi, len(agenti), dati.popolazione_totale
                )

            # 4f. Stampa riepilogo
            stampa_riepilogo(metriche, impianti_comune, stats)

            if analisi and "_sommario" in analisi:
                s = analisi["_sommario"]
                log.info(f"Analisi mode2: errore_medio={s['errore_medio_pct']}% "
                         f"valutazione={s['valutazione']}")

        except Exception as e:
            log.error(f"Errore per {codice_istat}: {e}", exc_info=True)
            if run_id and not args.no_db:
                segna_errore(run_id)

    log.info(f"\nCompletato in {round(time.time()-t_totale, 1)}s")


if __name__ == "__main__":
    main()
