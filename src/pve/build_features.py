"""Run every SQL file in sql/ in order with DuckDB and save each table."""
from pathlib import Path

import duckdb

RAW = Path("data/raw")
SQL = Path("sql")
OUT = Path("data/features")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()   # in-memory database: no server needed

    # make the raw Parquet files queryable as tables
    for name in ["players", "bets", "deposits"]:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM '{RAW / name}.parquet'")

    for path in sorted(SQL.glob("*.sql")):
        table = path.stem.split("_", 1)[1]       # 01_player_daily -> player_daily
        con.execute(path.read_text())
        con.execute(f"COPY {table} TO '{OUT / table}.parquet' (FORMAT parquet)")
        rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{path.name:<30} -> {table:<20} {rows:>10,} rows")


if __name__ == "__main__":
    main()
