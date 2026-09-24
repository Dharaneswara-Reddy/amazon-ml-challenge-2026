"""Tests for ber.io: exact TSV round trip into the Parquet cache, loud failures, idempotence."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import SOURCE_HEADER, write_text

from ber import io

# (raw TSV fields, expected decoded values)
S1_ROWS = [
    (
        ["S1-1", "Royal Food Pvt, Ltd", "12, MG Road, Bangalore", "India"],
        ["S1-1", "Royal Food Pvt, Ltd", "12, MG Road, Bangalore", "India"],
    ),
    (["S1-2", 'Joe"s Pizza', "", "US"], ["S1-2", 'Joe"s Pizza', "", "US"]),
    (
        ["S1-3", "राम मार्केटिंग प्राइवेट लिमिटेड", "C\\O RAM, नई दिल्ली", "India"],
        ["S1-3", "राम मार्केटिंग प्राइवेट लिमिटेड", "C\\O RAM, नई दिल्ली", "India"],
    ),
    (
        ["S1-4", '"""chrs & cie sasu"', '"4Th Floor, ""Niagara"" Building"', "France"],
        ["S1-4", '"chrs & cie sasu', '4Th Floor, "Niagara" Building', "France"],
    ),
    (["S1-5", "-- << Acme's", "a 'b'", ""], ["S1-5", "-- << Acme's", "a 'b'", ""]),
]


def source_tsv(rows: list[list[str]]) -> str:
    """Render source rows as TSV text with the organizer header."""
    return SOURCE_HEADER + "".join("\t".join(r) + "\n" for r in rows)


def make_dataset(root: Path) -> None:
    """Write all 7 organizer-shaped files with tiny content."""
    d = root / "dataset"
    for split in io.SPLITS:
        write_text(d / split / f"{split}_source1.tsv", source_tsv([r for r, _ in S1_ROWS]))
        write_text(
            d / split / f"{split}_source2.tsv",
            source_tsv([["S2-1", "A", "", "US"], ["S2-2", "B", "x", "US"]]),
        )
        write_text(d / split / f"{split}_source3.tsv", source_tsv([["S3-1", "C", "y", "India"]]))
    write_text(
        d / "train" / "train_ground_truth.tsv",
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\nS1-3\tS2-2\nS1-4\t\nS1-5\t\n",
    )


def test_read_round_trip_and_row_count(con, tmp_path: Path) -> None:
    """Commas, stray quote, CSV-quoted fields, empty fields, backslashes and Devanagari decode exactly."""
    src = write_text(tmp_path / "s1.tsv", source_tsv([r for r, _ in S1_ROWS]))
    out = tmp_path / "s1.parquet"
    stats = io.convert_source(con, src, out, 1)
    assert stats["rows"] == stats["expected_rows"] == len(S1_ROWS) == io.count_data_rows(src)
    assert stats["empty_address"] == 1
    got = con.execute(
        f"SELECT entity_id, business_name, business_address, country, source FROM read_parquet('{out}') "
        "ORDER BY entity_id"
    ).fetchall()
    assert [list(r[:4]) for r in got] == [exp for _, exp in S1_ROWS]
    assert {r[4] for r in got} == {1}


def test_no_trailing_newline(con, tmp_path: Path) -> None:
    """The last line without a newline still counts and converts."""
    src = write_text(tmp_path / "s.tsv", SOURCE_HEADER + "S2-1\ta\tb\tUS\nS2-2\tc\t\tIndia")
    assert io.convert_source(con, src, tmp_path / "o.parquet", 2)["rows"] == 2


@pytest.mark.parametrize(
    "body, match",
    [
        ("S1-1\ta\tb\tUS\textra\n", "Expected Number of Columns"),
        ("S1-1\ta\tb\n", "Expected Number of Columns"),
        ('S1-1\t"unterminated\tb\tUS\n', "malformed quoted field"),
        ('S1-1\t"a"b"\tb\tUS\n', "malformed quoted field"),
    ],
)
def test_bad_lines_fail_loudly(con, tmp_path: Path, body: str, match: str) -> None:
    """Wrong column counts and malformed quoting raise instead of being skipped or merged."""
    src = write_text(tmp_path / "bad.tsv", SOURCE_HEADER + "S1-0\tok\tok\tUS\n" + body)
    with pytest.raises(Exception, match=match):
        io.convert_source(con, src, tmp_path / "bad.parquet", 1)


def test_wrong_prefix_and_header(con, tmp_path: Path) -> None:
    """IDs from the wrong source and unexpected headers are rejected."""
    src = write_text(tmp_path / "s.tsv", SOURCE_HEADER + "S2-1\ta\tb\tUS\n")
    with pytest.raises(ValueError, match="prefix S1-"):
        io.convert_source(con, src, tmp_path / "o.parquet", 1)
    bad = write_text(tmp_path / "h.tsv", "id\tname\taddr\tcountry\nS1-1\ta\tb\tUS\n")
    with pytest.raises(ValueError, match="header"):
        io.convert_source(con, bad, tmp_path / "o.parquet", 1)


def test_prepare_full_and_idempotent(cfg, root: Path, con) -> None:
    """prepare writes all cache files, explodes GT, and skips fresh outputs on the second run."""
    make_dataset(root)
    first = io.prepare(cfg, con=con)
    assert first["raw_train_s1.parquet"]["rows"] == len(S1_ROWS)
    assert first["raw_test_s2.parquet"]["rows"] == 2
    gt = first["train_gt_pairs.parquet"]
    assert gt["rows"] == 5 and gt["pairs"] == 3 and gt["singletons"] == 3
    assert sorted(io.open_gt_pairs(con, cfg).fetchall()) == [
        ("S1-1", "S2-1"),
        ("S1-1", "S3-1"),
        ("S1-3", "S2-2"),
    ]
    assert dict(io.open_gt_s1(con, cfg).fetchall()) == {"S1-1": 2, "S1-2": 0, "S1-3": 1, "S1-4": 0, "S1-5": 0}
    assert io.open_right(con, cfg, "test").count("*").fetchone()[0] == 3

    second = io.prepare(cfg, con=con)
    assert all(v["skipped"] for v in second.values())

    src = io.raw_tsv_path(cfg, "train", 3)
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    third = io.prepare(cfg, con=con)
    assert third["raw_train_s3.parquet"]["skipped"] is False
    assert third["raw_train_s1.parquet"]["skipped"] is True
    meta = json.loads((io.cache_dir(cfg) / "raw_train_s3.parquet.meta.json").read_text())
    assert meta["rows"] == 1 and meta["format_version"] == io.IO_FORMAT_VERSION


def test_gt_rejects_bad_tokens(cfg, root: Path, con) -> None:
    """Empty tokens (double comma) or S1 ids inside a GT list fail loudly."""
    make_dataset(root)
    for body in ("S1-1\tS2-1,,S3-1\n", "S1-1\tS1-7\n"):
        write_text(root / "dataset/train/train_ground_truth.tsv", "source1_entity_id\tmatched_entity_ids\n" + body)
        with pytest.raises(ValueError, match="not S2-/S3-"):
            io.convert_ground_truth(con, io.gt_tsv_path(cfg), io.gt_pairs_path(cfg), io.gt_s1_path(cfg))
