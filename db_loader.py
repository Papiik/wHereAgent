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

    # ── Carica impianti da campagneresult ───────────────────────
    def carica_impianti(
        self,
        id_campagna: int,
        giorni_campagna: int = 14,
    ) -> list[Impianto]:
        """
        Legge tutti gli impianti di una campagna dalla tabella campagneresult.
        Usa TipoImpianto come base per la classificazione del tipo.
        """
        query = """
            SELECT
                id,
                CodiceInpe,
                TipoImpianto,
                TipoImpiantoDet,
                Formato,
                Latitudine,
                Longitudine,
                istatComune,
                idProvincia,
                Cimasa
            FROM campagneresult
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

            tipo = normalizza_tipo(r["TipoImpianto"])

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
                illuminato=True,
                digitale=(tipo == "dooh"),
                angolo_visibilita=90.0,
                velocita_media_kmh=velocita,
                giorni_campagna=giorni_campagna,
            ))

        print(f"[MySQLLoader] Caricati {len(impianti)} impianti per campagna {id_campagna}")
        return impianti

    # ── Carica zone da campagneresult ────────────────────────────
    def carica_zone(self, id_campagna: int) -> dict[str, ZonaDemografica]:
        """
        Legge i dati demografici dalla tabella campagneresult per i comuni
        degli impianti della campagna, usando istatComune come chiave.
        """
        query = """
            SELECT DISTINCT
                istatComune,
                comune,
                provincia,
                regione,
                TipoImpianto,
                abitanti,
                densita,
                superficie
            FROM campagneresult
            WHERE idCampagna = %s
              AND abitanti IS NOT NULL
              AND abitanti > 0
        """
        self.cursor.execute(query, (id_campagna,))
        rows = self.cursor.fetchall()

        zone = {}
        for r in rows:
            istat = r["istatComune"]
            if not istat or istat in zone:
                continue

            abitanti = int(r["abitanti"] or 0)
            if abitanti == 0:
                continue

            densita = float(r["densita"] or 0)
            if densita <= 0:
                densita = self._stima_densita(abitanti)

            flusso_ped_stimato = self._stima_flusso(abitanti, densita)

            zone[istat] = ZonaDemografica(
                zona=istat,
                popolazione=abitanti,
                densita_abitativa=densita,
                eta_media=42.0,
                pct_18_34=0.28,
                pct_35_54=0.37,
                pct_55plus=0.35,
                flusso_pedonale_ora_peak=flusso_ped_stimato,
                flusso_veicolare_ora_peak=int(flusso_ped_stimato * 0.6),
                ore_peak_giorno=6.5,
            )

        print(f"[MySQLLoader] Caricate {len(zone)} zone demografiche")
        return zone

    # ── Helpers privati ─────────────────────────────────────────
    def _stima_densita(self, abitanti: int) -> float:
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

    def _stima_flusso(self, abitanti: int, densita: float) -> int:
        """
        Stima il flusso pedonale orario in peak basandosi su popolazione
        e densità abitativa del comune.
        """
        if abitanti <= 0:
            return 100
        # Comuni più densi hanno maggiore mobilità pedonale
        coeff = min(densita / 1000, 3.0)  # cap a 3x
        flusso = int(abitanti * 0.04 * (1 + coeff) / 6)
        return max(flusso, 100)