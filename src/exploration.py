import duckdb
from pathlib import Path

RACINE = Path(__file__).resolve().parent 

con = duckdb.connect(str(RACINE / "rossmann.duckdb"))
con.execute(f"CREATE OR REPLACE VIEW train AS SELECT * FROM '{(RACINE / 'train.csv').as_posix()}'")
con.execute(f"CREATE OR REPLACE VIEW store AS SELECT * FROM '{(RACINE / 'store.csv').as_posix()}'")

sql = (RACINE / "sql" / "analyse.sql").read_text(encoding="utf-8")
requetes = [r.strip() for r in sql.split(";") if "select" in r.lower()]
print(f"{len(requetes)} requête(s) trouvée(s)")

for i, req in enumerate(requetes, 1):
    res = con.execute(req)
    print(f"--- Requête {i} ---")
    print(res.df().to_string())  
    print()
