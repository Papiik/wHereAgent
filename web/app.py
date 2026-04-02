"""
web/app.py — Interfaccia Flask per il simulatore ABM OOH.
"""
from __future__ import annotations
import sys
import os
import threading
import uuid
import logging
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, render_template, request, jsonify, send_file
import mysql.connector

import hashlib
import time

from abm.config import DB_CONFIG, SIM_WORKERS
from abm.campaign_loader import load_campaign
from abm.istat_loader import load_istat_data, arricchisci_coordinate
from abm.city_mapper import load_city_map, calc_dist_m
from abm.population_builder import build_population
from abm.simulation_engine import run_simulation
from abm.exposure_collector import aggrega_per_impianto
from abm.ppt_generator import genera_ppt
from abm.db_writer import crea_tabelle, apri_run, chiudi_run, segna_errore, salva_risultati

app = Flask(__name__)
log = logging.getLogger(__name__)

# job_id → { status, progress, message, result, error }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


# ── DB helpers ────────────────────────────────────────────────────────────────
def _get_clienti() -> list[dict]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT Cliente,
                   COUNT(DISTINCT idCampagna) as n_campagne
            FROM campagneresult
            WHERE Latitudine IS NOT NULL AND Latitudine != '0'
            GROUP BY Cliente
            ORDER BY Cliente
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        log.error(f"DB error: {e}")
        return []


def _get_campagne_cliente(cliente: str) -> list[dict]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT idCampagna, campagna, Cliente,
                   COUNT(*) as n_impianti,
                   COUNT(DISTINCT istatComune) as n_comuni
            FROM campagneresult
            WHERE Cliente = %s
              AND Latitudine IS NOT NULL AND Latitudine != '0'
            GROUP BY idCampagna, campagna, Cliente
            ORDER BY idCampagna DESC
        """, (cliente,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        log.error(f"DB error: {e}")
        return []


def _get_campagne() -> list[dict]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT idCampagna, campagna, Cliente,
                   COUNT(*) as n_impianti,
                   COUNT(DISTINCT istatComune) as n_comuni
            FROM campagneresult
            WHERE Latitudine IS NOT NULL AND Latitudine != '0'
            GROUP BY idCampagna, campagna, Cliente
            ORDER BY idCampagna DESC
            LIMIT 200
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        log.error(f"DB error: {e}")
        return []


def _get_comuni_campagna(id_campagna: int) -> list[dict]:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT DISTINCT istatComune, comune,
                   COUNT(*) as n_impianti
            FROM campagneresult
            WHERE idCampagna = %s
              AND istatComune IS NOT NULL
            GROUP BY istatComune, comune
            ORDER BY comune
        """, (id_campagna,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        log.error(f"DB error: {e}")
        return []


# ── Cache DB helpers ──────────────────────────────────────────────────────────
def _comuni_hash(comuni: list[str]) -> str:
    """Hash riproducibile della lista comuni (ordine insensitive)."""
    return hashlib.md5(",".join(sorted(comuni)).encode()).hexdigest()[:16]


def _cerca_cache(id_campagna: int, comuni: list[str], giorni: int, mode: int) -> dict | None:
    """
    Cerca un run completato con gli stessi parametri in abm_runs + abm_impianto_metrics.
    Ritorna dict {risultati, run_timestamp} oppure None.
    """
    h = _comuni_hash(comuni)
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        # Trova il run più recente con questi parametri
        cur.execute("""
            SELECT id, run_timestamp
            FROM abm_runs
            WHERE id_campagna = %s
              AND n_giorni    = %s
              AND profilo_mode = %s
              AND status      = 'done'
              AND JSON_UNQUOTE(JSON_EXTRACT(config_json, '$.comuni_hash')) = %s
            ORDER BY run_timestamp DESC
            LIMIT 1
        """, (id_campagna, giorni, mode, h))
        run = cur.fetchone()
        if not run:
            cur.close(); conn.close()
            return None

        run_id = run["id"]
        run_ts = str(run["run_timestamp"])

        # Leggi le metriche per impianto
        cur.execute("""
            SELECT impianto_id, ots_totali, reach_univoco,
                   frequency_media, grp, copertura_pct
            FROM abm_impianto_metrics
            WHERE run_id = %s
        """, (run_id,))
        rows = cur.fetchall()
        cur.close(); conn.close()

        if not rows:
            return None

        risultati = {
            r["impianto_id"]: {
                "ots_totali":      r["ots_totali"],
                "reach_univoco":   r["reach_univoco"],
                "frequency_media": r["frequency_media"],
                "grp":             r["grp"],
                "copertura_pct":   r["copertura_pct"],
            }
            for r in rows
        }
        return {"risultati": risultati, "run_timestamp": run_ts, "run_id": run_id}

    except Exception as e:
        log.warning(f"Cache lookup fallita: {e}")
        return None


def _salva_cache(id_campagna: int, comuni: list[str], giorni: int, mode: int,
                 nome_campagna: str, tutti_risultati: dict, tutti_eventi: list,
                 tutti_agenti: list, impianti: list, popolazione_totale: int,
                 durata_sec: float):
    """Salva i risultati simulazione su MySQL per uso futuro."""
    h = _comuni_hash(comuni)
    config_snap = {"comuni_hash": h, "comuni": comuni, "giorni": giorni, "mode": mode}
    try:
        crea_tabelle()
        run_id = apri_run(
            id_campagna=id_campagna,
            id_comune=",".join(comuni[:3]) + ("..." if len(comuni) > 3 else ""),
            nome_comune=nome_campagna,
            n_agenti=len(tutti_agenti),
            n_giorni=giorni,
            n_impianti=len(impianti),
            profilo_mode=mode,
            config_snapshot=config_snap,
        )
        salva_risultati(
            run_id=run_id,
            id_campagna=id_campagna,
            eventi=tutti_eventi,
            agenti=tutti_agenti,
            impianti=impianti,
            popolazione=popolazione_totale,
        )
        chiudi_run(run_id, durata_sec=round(durata_sec, 1), esposizioni=len(tutti_eventi))
        log.info(f"Risultati salvati in cache (run_id={run_id})")
    except Exception as e:
        log.warning(f"Salvataggio cache fallito: {e}")


# ── Simulation worker ─────────────────────────────────────────────────────────
def _run_job(job_id: str, params: dict):
    try:
        id_campagna = int(params["campagna"])
        scale       = float(params.get("scale") or 0.01)
        giorni      = int(params.get("giorni", 7))
        mode        = int(params.get("mode", 2))
        workers     = SIM_WORKERS
        comuni_sel  = params.get("comuni", [])   # lista istat, [] = tutti

        _set_job(job_id, progress=5, message="Caricamento campagna...")
        camp = load_campaign(id_campagna)

        imp_per_comune: dict[str, list] = defaultdict(list)
        for imp in camp["impianti"]:
            imp_per_comune[imp.istat_comune].append(imp)

        comuni_da_simulare = (
            [c for c in camp["comuni_istat"] if c in comuni_sel]
            if comuni_sel else camp["comuni_istat"]
        )

        impianti_dict = {imp.id: imp for imp in camp["impianti"]}

        # ── Controlla cache DB ───────────────────────────────────────────────
        _set_job(job_id, progress=8, message="Controllo cache risultati...")
        cache = _cerca_cache(id_campagna, comuni_da_simulare, giorni, mode)
        if cache:
            _set_job(job_id,
                     status="completed",
                     progress=100,
                     message="Risultati caricati dalla cache.",
                     result={
                         "campagna":      camp,
                         "risultati":     cache["risultati"],
                         "impianti":      impianti_dict,
                         "analisi":       None,
                         "mode":          mode,
                         "da_cache":      True,
                         "cache_timestamp": cache["run_timestamp"],
                     })
            return

        # ── Simulazione ──────────────────────────────────────────────────────
        _set_job(job_id, progress=10, message="Caricamento dati ISTAT...")
        dati_istat = load_istat_data(
            comuni_da_simulare, imp_per_comune, camp["comuni_info"]
        )
        arricchisci_coordinate(dati_istat, imp_per_comune)

        tutti_risultati: dict[str, dict] = {}
        tutti_eventi:    list            = []
        tutti_agenti:    list            = []
        popolazione_tot: int             = 0
        analisi_globale: dict | None     = None
        n = len(comuni_da_simulare)
        t_start = time.time()

        for i, istat in enumerate(comuni_da_simulare):
            dati = dati_istat.get(istat)
            if not dati:
                continue
            impianti_comune = imp_per_comune.get(istat, [])
            if not impianti_comune:
                continue
            lat, lon = dati.coordinate_centroide
            if not lat or not lon:
                continue

            pct = 15 + int((i / n) * 75)
            _set_job(job_id, progress=pct,
                     message=f"Simulazione {dati.nome_comune} ({i+1}/{n})...")

            dist_m   = calc_dist_m(lat, lon, impianti_comune)
            city_map = load_city_map(istat, dati.nome_comune, lat, lon, dist_m=dist_m)
            agenti, analisi_comune = build_population(dati, city_map, mode=mode)
            if analisi_comune and not analisi_globale:
                analisi_globale = analisi_comune
            eventi, _ = run_simulation(
                agenti, city_map, impianti_comune,
                n_giorni=giorni, n_workers=workers,
            )
            metriche = aggrega_per_impianto(
                eventi, len(agenti), dati.popolazione_totale
            )
            tutti_risultati.update(metriche)
            tutti_eventi.extend(eventi)
            tutti_agenti.extend(agenti)
            popolazione_tot += dati.popolazione_totale

        _set_job(job_id, progress=92, message="Salvataggio risultati su DB...")
        _salva_cache(
            id_campagna=id_campagna,
            comuni=comuni_da_simulare,
            giorni=giorni,
            mode=mode,
            nome_campagna=camp.get("nome_campagna", ""),
            tutti_risultati=tutti_risultati,
            tutti_eventi=tutti_eventi,
            tutti_agenti=tutti_agenti,
            impianti=camp["impianti"],
            popolazione_totale=popolazione_tot,
            durata_sec=time.time() - t_start,
        )

        _set_job(job_id, progress=95, message="Preparazione risultati...")

        _set_job(job_id,
                 status="completed",
                 progress=100,
                 message="Simulazione completata.",
                 result={
                     "campagna":  camp,
                     "risultati": tutti_risultati,
                     "impianti":  impianti_dict,
                     "analisi":   analisi_globale,
                     "mode":      mode,
                     "da_cache":  False,
                 })

    except Exception as e:
        log.exception("Errore simulazione")
        _set_job(job_id, status="error", message=str(e))


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    clienti = _get_clienti()
    return render_template("index.html", clienti=clienti)


@app.route("/api/clienti")
def api_clienti():
    return jsonify(_get_clienti())


@app.route("/api/campagne")
def api_campagne():
    cliente = request.args.get("cliente", "")
    if cliente:
        return jsonify(_get_campagne_cliente(cliente))
    return jsonify(_get_campagne())


@app.route("/api/comuni/<int:id_campagna>")
def api_comuni(id_campagna: int):
    return jsonify(_get_comuni_campagna(id_campagna))


@app.route("/api/run", methods=["POST"])
def api_run():
    params = request.json or {}
    job_id = uuid.uuid4().hex[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "status":   "running",
            "progress": 0,
            "message":  "Avvio...",
            "result":   None,
            "error":    None,
        }
    t = threading.Thread(target=_run_job, args=(job_id, params), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job non trovato"}), 404

    out = {k: v for k, v in job.items() if k != "result"}

    if job["status"] == "completed" and job["result"]:
        risultati = job["result"]["risultati"]
        impianti  = job["result"]["impianti"]
        out["da_cache"]        = job["result"].get("da_cache", False)
        out["cache_timestamp"] = job["result"].get("cache_timestamp", "")
        out["mode"]            = job["result"].get("mode", 1)
        out["analisi"]         = job["result"].get("analisi")
        out["table"] = [
            {
                "id":        imp_id,
                "tipo":      impianti[imp_id].tipo if imp_id in impianti else "",
                "indirizzo": impianti[imp_id].indirizzo[:40] if imp_id in impianti else "",
                "lat":       impianti[imp_id].lat if imp_id in impianti else None,
                "lon":       impianti[imp_id].lon if imp_id in impianti else None,
                **m,
            }
            for imp_id, m in sorted(
                risultati.items(),
                key=lambda x: x[1]["ots_totali"],
                reverse=True,
            )
        ]
    return jsonify(out)


@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "completed":
        return "Job non completato", 400

    path = genera_ppt(job["result"])
    id_c = job["result"]["campagna"].get("id_campagna", "x")
    return send_file(
        path,
        as_attachment=True,
        download_name=f"report_ooh_{id_c}.pptx",
        mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # use_reloader=False necessario su Windows: il reloader interferisce
    # con ProcessPoolExecutor (multiprocessing) nei thread di simulazione
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
