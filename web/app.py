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

from abm.config import DB_CONFIG, SIM_WORKERS
from abm.campaign_loader import load_campaign
from abm.istat_loader import load_istat_data, arricchisci_coordinate
from abm.city_mapper import load_city_map, calc_dist_m
from abm.population_builder import build_population
from abm.simulation_engine import run_simulation
from abm.exposure_collector import aggrega_per_impianto
from abm.ppt_generator import genera_ppt

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


# ── Simulation worker ─────────────────────────────────────────────────────────
def _run_job(job_id: str, params: dict):
    try:
        id_campagna = int(params["campagna"])
        scale       = float(params.get("scale", 0.01))
        giorni      = int(params.get("giorni", 7))
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

        _set_job(job_id, progress=10, message="Caricamento dati ISTAT...")
        dati_istat = load_istat_data(
            comuni_da_simulare, imp_per_comune, camp["comuni_info"]
        )
        arricchisci_coordinate(dati_istat, imp_per_comune)

        tutti_risultati: dict[str, dict] = {}
        n = len(comuni_da_simulare)

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
            agenti, _ = build_population(dati, city_map, mode=1)
            eventi, _ = run_simulation(
                agenti, city_map, impianti_comune,
                n_giorni=giorni, n_workers=workers,
            )
            metriche = aggrega_per_impianto(
                eventi, len(agenti), dati.popolazione_totale
            )
            tutti_risultati.update(metriche)

        _set_job(job_id, progress=95, message="Preparazione risultati...")

        impianti_dict = {imp.id: imp for imp in camp["impianti"]}

        _set_job(job_id,
                 status="completed",
                 progress=100,
                 message="Simulazione completata.",
                 result={
                     "campagna":  camp,
                     "risultati": tutti_risultati,
                     "impianti":  impianti_dict,
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
