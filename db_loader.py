# db_loader.py
import mysql.connector
from ooh_agents import Impianto, ZonaDemografica

# ── Mapping tipo impianto DB → tipo modello ──────────────────
TIPO_MAP = {
    "billboard":   "billboard",
    "poster":      "billboard",
    "pensilina":   "pensilina",
    "fermata":     "pensilina",
    "metro":       "metro",
    "metropolitana": "metro",
    "digital":     "dooh",
    "dooh":        "dooh",
    "led":         "dooh",
}

def normalizza_tipo(tipo_raw: str) -> str:
    """Mappa il TipoImpiantoDesc del DB al tipo del modello."""
    if not tipo_raw:
        return "billboard"
    t = tipo_raw.lower().strip()
    for chiave, valore in TIPO_MAP.items():
        if chiave in t:
            return valore
    return "billboard"  # fallback


class MySQLLoader:
    """Carica impianti e zone demografiche dal database."""

    def __init__(self, host: str, user: str, password: str, database: str, port: int = 3306):
        self.conn = mysql.connector.connect(
            host=host, user=user, password=password,
            database=database, port=port,
            connection_timeout=10,
        )
        self.cursor = self.conn.cursor(dictionary=True)

    def close(self):
        self.cursor.close()
        self.conn.close()

    # ── Carica impianti da campaigns ────────────────────────────
    def carica_impianti(
        self,
        id_campagna: int,
        giorni_campagna: int = 14,
    ) -> list[Impianto]:
        """
        Legge tutti gli impianti di una campagna dalla tabella campaigns.
        """
        query = """
            SELECT
                id,
                CodiceInpe,
                TipoImpiantoDesc,
                TipoImpiantoDet,
                Formato,
                Latitudine,
                Longitudine,
                istatComune,
                idProvincia,
                Cimasa
            FROM campaigns
            WHERE idCampagna = %s
              AND Latitudine IS NOT NULL
              AND Longitudine IS NOT NULL
        """
        self.cursor.execute(query, (id_campagna,))
        rows = self.cursor.fetchall()

        impianti = []
        for r in rows:
            try:
                lat = float(r["Latitudine"].replace(",", "."))
                lon = float(r["Longitudine"].replace(",", "."))
            except (ValueError, AttributeError):
                continue  # salta impianti senza coordinate valide

            tipo = normalizza_tipo(r["TipoImpiantoDesc"])

            # Stima velocità in base al tipo
            velocita = {
                "billboard": 40, "pensilina": 5,
                "metro": 2, "dooh": 30,
            }.get(tipo, 30)

            # Zona = codice istat comune (usato come chiave per le zone demo)
            zona_key = r["istatComune"] or f"prov_{r['idProvincia']}"

            impianti.append(Impianto(
                id=str(r["id"]),
                tipo=tipo,
                lat=lat,
                lon=lon,
                zona=zona_key,
                formato=r["Formato"] or "6x3",
                illuminato=True,        # aggiorna se hai il campo in DB
                digitale=(tipo == "dooh"),
                angolo_visibilita=90.0,
                velocita_media_kmh=velocita,
                giorni_campagna=giorni_campagna,
            ))

        print(f"[MySQLLoader] Caricati {len(impianti)} impianti per campagna {id_campagna}")
        return impianti

    # ── Carica zone da view_deepooh ──────────────────────────────
    def carica_zone(self, id_campagna: int) -> dict[str, ZonaDemografica]:
        """
        Legge i dati demografici dalla view_deepooh per i comuni
        degli impianti della campagna, usando istat come chiave.

        Nota: view_deepooh ha già grpAdu e abitanti — li usiamo
        come riferimento per calibrare i flussi stimati.
        """
        query = """
            SELECT DISTINCT
                v.istat,
                v.comune,
                v.provincia,
                v.Regione,
                v.TipoImpianto,
                v.abitanti,
                v.grpAdu,
                v.grpGravAdu,
                v.grpNazAdu
            FROM view_deepooh v
            INNER JOIN campaigns c
                ON v.istat = c.istatComune
            WHERE c.idCampagna = %s
              AND v.abitanti IS NOT NULL
              AND v.abitanti > 0 limit 5
        """
        self.cursor.execute(query, (id_campagna,))
        rows = self.cursor.fetchall()

        zone = {}
        for r in rows:
            istat = r["istat"]
            if not istat or istat in zone:
                continue

            abitanti = int(r["abitanti"] or 0)
            if abitanti == 0:
                continue

            # Stima densità da popolazione (approssimazione)
            # In assenza di dati superficie, uso fasce per regione
            densita_stimata = self._stima_densita(r["comune"], r["Regione"], abitanti)

            # Stima flussi dal GRP già presente in DB
            # grpAdu = GRP adulti certificato → backsolve flusso pedonale
            grp_adu = float(r["grpAdu"] or 0)
            flusso_ped_stimato = self._backsolve_flusso(grp_adu, abitanti)

            zone[istat] = ZonaDemografica(
                zona=istat,
                popolazione=abitanti,
                densita_abitativa=densita_stimata,
                eta_media=42.0,      # media Italia; aggiorna con ISTAT per comune
                pct_18_34=0.28,      # medie nazionali ISTAT 2023
                pct_35_54=0.37,
                pct_55plus=0.35,
                flusso_pedonale_ora_peak=flusso_ped_stimato,
                flusso_veicolare_ora_peak=int(flusso_ped_stimato * 0.6),
                ore_peak_giorno=6.5,
            )

        print(f"[MySQLLoader] Caricate {len(zone)} zone demografiche")
        return zone

    # ── Helpers privati ─────────────────────────────────────────
    def _stima_densita(self, comune: str, regione: str, abitanti: int) -> float:
        """Stima la densità abitativa in base alla dimensione del comune."""
        if abitanti > 500000:
            return 7000.0
        elif abitanti > 100000:
            return 3500.0
        elif abitanti > 50000:
            return 1500.0
        elif abitanti > 10000:
            return 600.0
        else:
            return 200.0

    def _backsolve_flusso(self, grp_adu: float, abitanti: int) -> int:
        """
        Backsolve del flusso pedonale dal GRP certificato.
        GRP = (Reach/Pop)*100 * Frequency
        Assumendo VI=0.85, AF=0.45, CT=1.0, k=0.50, giorni=14:
        si può stimare il flusso giornaliero approssimativo.
        """
        if grp_adu <= 0 or abitanti <= 0:
            return max(int(abitanti * 0.03), 100)  # fallback: 3% pop/ora

        # Stima conservativa: grpAdu è già un output, lo usiamo come
        # proxy per calibrare il flusso base
        flusso_stimato = int((grp_adu / 100) * abitanti / 14 / 6)
        return max(flusso_stimato, 100)