"""Tests for ber.features on a tiny synthetic world."""

from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from ber.blocking import BlockingConfig, register_inputs, run_blocking
from ber.features import FEATURES, write_features

COLS = "entity_id, name_norm, name_core, addr_norm, postal, addr_nums, script, country_norm"
LEFT = [
    (
        "S1-1",
        "ram marketing pvt ltd",
        "ram marketing",
        "12 mg road bangalore 560001",
        "560001",
        "12",
        "latin",
        "india",
    ),
    ("S1-2", "west siddhi corp", "west siddhi", "3109 sunshine drive clinton md", "", "3109", "latin", "us"),
]
RIGHT = [
    (
        "S2-1",
        "ram marketing private limited",
        "ram marketing",
        "12 m g road bengaluru 560001",
        "560001",
        "12",
        "latin",
        "india",
    ),
    ("S3-1", "west siddhi", "west siddhi", "", "", "", "latin", "us"),
    ("S3-2", "ram textiles", "ram textiles", "44 ring road delhi 110001", "110001", "44", "latin", "india"),
]


def _write(con: duckdb.DuckDBPyConnection, rows: list[tuple], path: Path) -> None:
    """Write normalized rows to parquet."""
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE t AS SELECT * FROM (VALUES {', '.join(map(str, rows))}) v({COLS})"
    )
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")


def test_feature_matrix_shape_and_labels(tmp_path: Path) -> None:
    """Every candidate gets every feature, labels mark true pairs, similarities stay in [0, 1]."""
    con = duckdb.connect()
    _write(con, LEFT, tmp_path / "l.parquet")
    _write(con, RIGHT, tmp_path / "r.parquet")
    register_inputs(con, str(tmp_path / "l.parquet"), [str(tmp_path / "r.parquet")])
    run_blocking(con, BlockingConfig(sn_window=2, final_top_k=3, buckets=1))
    n_cand = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    gt = "SELECT * FROM (VALUES ('S1-1','S2-1'), ('S1-2','S3-1')) g(s1_id, match_id)"
    out = tmp_path / "f.parquet"
    n = write_features(con, out, gt, batch_rows=2, workers=1)
    tbl = pq.read_table(out)
    assert n == n_cand == tbl.num_rows
    assert set(FEATURES) <= set(tbl.column_names)
    rows = {(r["lid"], r["rid"]): r for r in tbl.to_pylist()}
    assert rows[("S1-1", "S2-1")]["label"] == 1
    assert rows[("S1-1", "S2-1")]["postal_state"] == 1
    assert rows[("S1-2", "S3-1")]["r_addr_empty"] == 1
    if ("S1-1", "S3-2") in rows:
        assert rows[("S1-1", "S3-2")]["label"] == 0
        assert rows[("S1-1", "S3-2")]["postal_state"] == -1
    for r in rows.values():
        for name in ("core_jw", "addr_token_set", "name_ratio", "tok_jacc_name"):
            assert 0.0 <= r[name] <= 1.0
