"""
Run the SQL feature pipeline with DuckDB.

Every file in sql/ runs in name order (01_, 02_, ...). Each file creates
one table, and each table is saved to data/features/<table>.parquet so the
models and notebooks can read it.

Usage:
    python -m pve.build_features
"""
from pathlib import Path

import duckdb

RAW = Path("data/raw")
SQL = Path(__file__).resolve().parents[2] / "sql"   # the repo's sql/ folder
OUT = Path("data/features")
SNAPSHOT_DATE = "2025-06-30"     # features use data up to here; labels come after


def connect(raw: Path = RAW, snapshot: str = SNAPSHOT_DATE) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with the raw tables and the snapshot date set."""
    con = duckdb.connect()   # in-memory database: no server needed
    for name in ["players", "bets", "deposits", "sof_checks"]:
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM '{raw / name}.parquet'")
    con.execute(f"SET VARIABLE snapshot_date = DATE '{snapshot}'")
    # one thread = numbers are always added in the same order, so every build
    # gives identical results to the last decimal place (parallel adding doesn't)
    con.execute("SET threads = 1")
    return con


def build(raw: Path = RAW, out: Path = OUT, verbose: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    con = connect(raw)

    for path in sorted(SQL.glob("*.sql")):
        table = path.stem.split("_", 1)[1]       # 01_player_daily -> player_daily
        con.execute(path.read_text())
        # DuckDB works in parallel, so row order isn't guaranteed: sort before
        # saving so every build writes an identical file (and models train identically)
        con.execute(f"COPY (SELECT * FROM {table} ORDER BY ALL) TO '{out / table}.parquet' (FORMAT parquet)")
        rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if verbose:
            print(f"{path.name:<30} -> {table:<20} {rows:>10,} rows")


if __name__ == "__main__":
    build()
