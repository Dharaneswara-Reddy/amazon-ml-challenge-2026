"""Tests for ber.blocking on a tiny synthetic world."""

from pathlib import Path

import duckdb
import pytest

from ber.blocking import BlockingConfig, estimate_token_pairs, recall_by_pass, register_inputs, run_blocking

LEFT = [
    ("S1-1", "ram marketing", "india"),
    ("S1-2", "west siddhi", "us"),
    ("S1-3", "blueridge medical centre", "india"),
    ("S1-4", "pearla king", "us"),
    ("S1-5", "sharma traders", "india"),
]
RIGHT = [
    ("S2-1", "ram marketing", "india"),
    ("S3-1", "ram marketng", "india"),
    ("S2-2", "west sidhi", "us"),
    ("S3-2", "medical centre blueridge", "india"),
    ("S2-3", "king pearla", "us"),
    ("S3-3", "sharma textiles", "india"),
    ("S2-4", "ram marketing", "us"),
]
GT = [("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2"), ("S1-3", "S3-2"), ("S1-4", "S2-3")]


def _write(con: duckdb.DuckDBPyConnection, rows: list[tuple], path: Path) -> None:
    """Write (entity_id, name_core, country_norm) rows to parquet."""
    con.execute("CREATE OR REPLACE TEMP TABLE t (entity_id VARCHAR, name_core VARCHAR, country_norm VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")


@pytest.fixture()
def con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with views L and R over the synthetic parquet files."""
    c = duckdb.connect()
    _write(c, LEFT, tmp_path / "l.parquet")
    _write(c, [r for r in RIGHT if r[0].startswith("S2")], tmp_path / "r2.parquet")
    _write(c, [r for r in RIGHT if r[0].startswith("S3")], tmp_path / "r3.parquet")
    register_inputs(
        c, str(tmp_path / "l.parquet"), [str(tmp_path / "r2.parquet"), str(tmp_path / "r3.parquet")]
    )
    return c


def test_blocking_finds_all_true_pairs(con: duckdb.DuckDBPyConnection) -> None:
    """Typos, word order and reordered tokens are all recovered on the toy world."""
    run_blocking(con, BlockingConfig(sn_window=2, final_top_k=3, buckets=2))
    gt_sql = "SELECT * FROM (VALUES " + ", ".join(f"('{a}','{b}')" for a, b in GT) + ") g(s1_id, match_id)"
    stats = recall_by_pass(con, gt_sql, ["tok", "sn_plain", "sn_rev", "sn_sorted"])
    assert stats["recall_final"] == 1.0


def test_candidates_never_cross_country(con: duckdb.DuckDBPyConnection) -> None:
    """A US right record is never paired with an India left record."""
    run_blocking(con, BlockingConfig(sn_window=2, final_top_k=3, buckets=2))
    crossed = con.execute(
        """SELECT count(*) FROM candidates c JOIN L l ON l.entity_id = c.lid JOIN R r ON r.entity_id = c.rid
           WHERE l.country_norm <> r.country_norm"""
    ).fetchone()[0]
    assert crossed == 0


def test_per_right_cap_bounds_output(con: duckdb.DuckDBPyConnection) -> None:
    """No right record has more than final_top_k candidates."""
    run_blocking(con, BlockingConfig(sn_window=3, final_top_k=1, buckets=2))
    worst = con.execute(
        "SELECT max(n) FROM (SELECT rid, count(*) n FROM candidates GROUP BY rid)"
    ).fetchone()[0]
    assert worst == 1


def test_estimate_token_pairs_monotone_in_cap(con: duckdb.DuckDBPyConnection) -> None:
    """A higher document-frequency cap never produces fewer token pairs."""
    run_blocking(con, BlockingConfig(buckets=1))
    est = estimate_token_pairs(con, [1, 2, 10])
    assert [e["pairs"] for e in est] == sorted(e["pairs"] for e in est)
