"""
DuckDB and PySpark must produce IDENTICAL features.

The SQL pipeline (DuckDB, one machine) and the PySpark pipeline (many
machines) are two implementations of the same logic. This test runs both
on the same small dataset and compares every value in every table.

Skipped automatically where PySpark or Java isn't available (e.g. Python
3.15 on a laptop). It always runs in Docker and in GitHub Actions.
"""
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java is needed for Spark", allow_module_level=True)

from pve import build_features, spark_features            # noqa: E402
from pve.generate_data import generate, save               # noqa: E402

KEYS = {
    "player_daily": ["player_id", "bet_date"],
    "rolling_features": ["player_id", "bet_date"],
    "bet_patterns": ["player_id", "bet_date"],
    "player_features": ["player_id"],
}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("parity")
    save(root / "raw", *generate(600, seed=5))
    build_features.build(root / "raw", root / "duckdb", verbose=False)
    spark = spark_features.spark_session()
    try:
        spark_features.build(spark, root / "raw", root / "spark")
    finally:
        spark.stop()
    return root


@pytest.mark.parametrize("table", list(KEYS))
def test_spark_matches_duckdb(outputs, table):
    keys = KEYS[table]
    duck = pd.read_parquet(outputs / "duckdb" / f"{table}.parquet").sort_values(keys).reset_index(drop=True)
    spark = pq.read_table(outputs / "spark" / table).to_pandas().sort_values(keys).reset_index(drop=True)

    assert list(duck.columns) == list(spark.columns), "same columns, same order"
    assert len(duck) == len(spark), "same number of rows"
    for col in duck.columns:
        a, b = duck[col], spark[col]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            # tiny float differences from adding in a different order are fine
            same = np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            same = (a.astype(str) == b.astype(str)).to_numpy()
        assert same.all(), f"{table}.{col}: {(~same).sum()} rows differ"
