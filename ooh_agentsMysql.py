from db_loader import MySQLLoader
from ooh_agents import OrchestratorAgent

# Connessione
loader = MySQLLoader(
    host="localhost",
    user="root",
    password="",
    database="dbooh",
)

# Carica dati per la campagna con idCampagna=42
impianti = loader.carica_impianti(id_campagna=42, giorni_campagna=14)
zone     = loader.carica_zone(id_campagna=42)
loader.close()

# Pipeline identica a prima
orc = OrchestratorAgent(verboso=True)
df  = orc.esegui_campagna(impianti, zone, target_eta="18_34")
agg = orc.aggrega_campagna(df, zone)

print(df[["impianto_id", "reach_univoco", "frequency_media", "grp"]])
print(agg)
df.to_csv("risultati.csv", index=False)