"""
db_writer.py — Crea le tabelle ABM su MySQL e scrive i risultati.

Tabelle:
  abm_runs                  — metadati simulazione
  abm_impianto_metrics      — OTS, Reach, GRP per impianto
  abm_reach_oraria          — reach per fascia oraria
  abm_reach_mezzo           — reach per mezzo di trasporto
  abm_frequenza_individuale — frequenza reale per agente-impianto
"""
from __future__ import annotations
import json
import logging
from datetime import datetime

import mysql.connector

from .config import DB_CONFIG
from .exposure_collector import (
    EsposizioneEvento,
    aggrega_per_impianto,
    aggrega_per_ora,
    aggrega_per_mezzo,
    aggrega_frequenza_individuale,
)
from .population_builder import AgentProfile
from .campaign_loader import ImpiantoABM

log = logging.getLogger(__name__)

# ── DDL tabelle ───────────────────────────────────────────────────────────────
_DDL = [
    """
    CREATE TABLE IF NOT EXISTS abm_runs (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        id_campagna   INT NOT NULL,
        id_comune     VARCHAR(10),
        nome_comune   VARCHAR(100),
        run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        n_agenti      INT,
        n_giorni      INT,
        n_impianti    INT,
        profilo_mode  TINYINT COMMENT '0=Claude 1=Python 2=Entrambi',
        config_json   TEXT,
        durata_sec    FLOAT,
        esposizioni   INT,
        status        VARCHAR(20) DEFAULT 'running',
        INDEX idx_campagna (id_campagna)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS abm_impianto_metrics (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        run_id           INT NOT NULL,
        impianto_id      VARCHAR(50) NOT NULL,
        id_campagna      INT NOT NULL,
        tipo             VARCHAR(30),
        ots_totali       INT,
        reach_univoco    INT,
        frequency_media  FLOAT,
        grp              FLOAT,
        copertura_pct    FLOAT,
        INDEX idx_run (run_id),
        INDEX idx_impianto (impianto_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS abm_reach_oraria (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        run_id          INT NOT NULL,
        impianto_id     VARCHAR(50) NOT NULL,
        ora             TINYINT,
        n_esposizioni   INT,
        n_agenti_unici  INT,
        INDEX idx_run_imp (run_id, impianto_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS abm_reach_mezzo (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        run_id          INT NOT NULL,
        impianto_id     VARCHAR(50) NOT NULL,
        mezzo           VARCHAR(20),
        n_esposizioni   INT,
        n_agenti_unici  INT,
        INDEX idx_run_imp (run_id, impianto_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS abm_frequenza_individuale (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        run_id          INT NOT NULL,
        agent_id        VARCHAR(80) NOT NULL,
        impianto_id     VARCHAR(50) NOT NULL,
        n_esposizioni   INT,
        INDEX idx_run (run_id),
        INDEX idx_agent (agent_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


def crea_tabelle(conn_params: dict = None):
    """Crea le tabelle ABM se non esistono."""
    params = conn_params or DB_CONFIG
    conn   = mysql.connector.connect(**params)
    cur    = conn.cursor()
    for ddl in _DDL:
        cur.execute(ddl)
    conn.commit()
    cur.close()
    conn.close()
    log.info("Tabelle ABM verificate/create.")


# ── Scrittura run ─────────────────────────────────────────────────────────────
def apri_run(
    id_campagna: int,
    id_comune: str,
    nome_comune: str,
    n_agenti: int,
    n_giorni: int,
    n_impianti: int,
    profilo_mode: int,
    config_snapshot: dict,
    conn_params: dict = None,
) -> int:
    """Inserisce una riga in abm_runs e ritorna il run_id."""
    params = conn_params or DB_CONFIG
    conn   = mysql.connector.connect(**params)
    cur    = conn.cursor()
    cur.execute("""
        INSERT INTO abm_runs
            (id_campagna, id_comune, nome_comune, n_agenti, n_giorni,
             n_impianti, profilo_mode, config_json, status)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'running')
    """, (
        id_campagna, id_comune, nome_comune,
        n_agenti, n_giorni, n_impianti, profilo_mode,
        json.dumps(config_snapshot, ensure_ascii=False),
    ))
    run_id = cur.lastrowid
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Run aperto: id={run_id}")
    return run_id


def chiudi_run(run_id: int, durata_sec: float,
               esposizioni: int, conn_params: dict = None):
    """Aggiorna lo stato del run a 'done'."""
    params = conn_params or DB_CONFIG
    conn   = mysql.connector.connect(**params)
    cur    = conn.cursor()
    cur.execute("""
        UPDATE abm_runs
        SET status='done', durata_sec=%s, esposizioni=%s
        WHERE id=%s
    """, (durata_sec, esposizioni, run_id))
    conn.commit()
    cur.close()
    conn.close()


def segna_errore(run_id: int, conn_params: dict = None):
    params = conn_params or DB_CONFIG
    conn   = mysql.connector.connect(**params)
    cur    = conn.cursor()
    cur.execute("UPDATE abm_runs SET status='error' WHERE id=%s", (run_id,))
    conn.commit()
    cur.close()
    conn.close()


# ── Scrittura metriche ────────────────────────────────────────────────────────
def _batch_insert(cur, sql: str, rows: list[tuple], batch: int = 500):
    for i in range(0, len(rows), batch):
        cur.executemany(sql, rows[i:i + batch])


def salva_risultati(
    run_id: int,
    id_campagna: int,
    eventi: list[EsposizioneEvento],
    agenti: list[AgentProfile],
    impianti: list[ImpiantoABM],
    popolazione: int,
    conn_params: dict = None,
):
    """
    Aggrega gli eventi e scrive tutte le tabelle di output.
    """
    params = conn_params or DB_CONFIG
    n_agenti = len(agenti)

    imp_dict = {imp.id: imp for imp in impianti}

    # ── 1. Metriche per impianto ─────────────────────────────────────────────
    metriche = aggrega_per_impianto(eventi, n_agenti, popolazione)

    # ── 2. Reach oraria ──────────────────────────────────────────────────────
    reach_ora = aggrega_per_ora(eventi)

    # ── 3. Reach per mezzo ───────────────────────────────────────────────────
    reach_mezzo = aggrega_per_mezzo(eventi)

    # ── 4. Frequenza individuale ──────────────────────────────────────────────
    freq_ind = aggrega_frequenza_individuale(eventi)

    conn = mysql.connector.connect(**params)
    cur  = conn.cursor()

    # Impianto metrics
    rows_imp = [
        (
            run_id, imp_id, id_campagna,
            imp_dict.get(imp_id, type("", (), {"tipo": "unknown"})()).tipo
            if imp_id in imp_dict else "unknown",
            m["ots_totali"], m["reach_univoco"],
            m["frequency_media"], m["grp"], m["copertura_pct"],
        )
        for imp_id, m in metriche.items()
    ]
    # Fix accesso tipo impianto
    rows_imp = []
    for imp_id, m in metriche.items():
        tipo = imp_dict[imp_id].tipo if imp_id in imp_dict else "unknown"
        rows_imp.append((
            run_id, imp_id, id_campagna, tipo,
            m["ots_totali"], m["reach_univoco"],
            m["frequency_media"], m["grp"], m["copertura_pct"],
        ))

    _batch_insert(cur, """
        INSERT INTO abm_impianto_metrics
            (run_id, impianto_id, id_campagna, tipo,
             ots_totali, reach_univoco, frequency_media, grp, copertura_pct)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, rows_imp)

    # Reach oraria
    rows_ora = [
        (run_id, r["impianto_id"], r["ora"],
         r["n_esposizioni"], r["n_agenti_unici"])
        for r in reach_ora
    ]
    _batch_insert(cur, """
        INSERT INTO abm_reach_oraria
            (run_id, impianto_id, ora, n_esposizioni, n_agenti_unici)
        VALUES (%s,%s,%s,%s,%s)
    """, rows_ora)

    # Reach mezzo
    rows_mez = [
        (run_id, r["impianto_id"], r["mezzo"],
         r["n_esposizioni"], r["n_agenti_unici"])
        for r in reach_mezzo
    ]
    _batch_insert(cur, """
        INSERT INTO abm_reach_mezzo
            (run_id, impianto_id, mezzo, n_esposizioni, n_agenti_unici)
        VALUES (%s,%s,%s,%s,%s)
    """, rows_mez)

    # Frequenza individuale
    rows_fi = [
        (run_id, r["agent_id"], r["impianto_id"], r["n_esposizioni"])
        for r in freq_ind
    ]
    _batch_insert(cur, """
        INSERT INTO abm_frequenza_individuale
            (run_id, agent_id, impianto_id, n_esposizioni)
        VALUES (%s,%s,%s,%s)
    """, rows_fi)

    conn.commit()
    cur.close()
    conn.close()

    log.info(
        f"Salvato run {run_id}: "
        f"{len(rows_imp)} impianti | "
        f"{len(rows_ora)} slot orari | "
        f"{len(rows_mez)} slot mezzo | "
        f"{len(rows_fi)} freq individuali"
    )
    return metriche


# ── Stampa riepilogo ──────────────────────────────────────────────────────────
def stampa_riepilogo(metriche: dict, impianti: list[ImpiantoABM], stats: dict):
    imp_dict = {imp.id: imp for imp in impianti}
    print("\n" + "=" * 70)
    print("  RISULTATI SIMULAZIONE ABM")
    print("=" * 70)
    print(f"  Agenti simulati : {stats['n_agenti']:,}")
    print(f"  Giorni          : {stats['n_giorni']}")
    print(f"  Slot totali     : {stats['slot_totali']:,}")
    print(f"  Esposizioni     : {stats['esposizioni']:,}")
    print(f"  Durata          : {stats['durata_sec']}s")
    print(f"  Velocita        : {stats['slot_per_sec']:,} slot/sec")
    print()
    print(f"  {'Impianto':<12} {'Tipo':<12} {'OTS':>8} "
          f"{'Reach':>8} {'Freq':>6} {'GRP':>7} {'Cop%':>6}")
    print("  " + "-" * 60)
    for imp_id, m in sorted(metriche.items(),
                            key=lambda x: -x[1]["reach_univoco"])[:20]:
        tipo = imp_dict.get(imp_id, type("",(),{"tipo":"?"})()).tipo \
               if imp_id in imp_dict else "?"
        print(f"  {imp_id:<12} {tipo:<12} {m['ots_totali']:>8,} "
              f"{m['reach_univoco']:>8,} {m['frequency_media']:>6.1f} "
              f"{m['grp']:>7.1f} {m['copertura_pct']:>6.2f}%")
    print("=" * 70)
