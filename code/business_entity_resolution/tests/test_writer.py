"""Tests for ber.writer: valid output passes the official validator; each violation raises."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from conftest import SOURCE_HEADER, write_text

from ber.config import PACKAGE_DIR
from ber.writer import write_outputs

VALIDATOR = PACKAGE_DIR.parents[1] / "student_resource" / "utils" / "validate_submission.py"
S1 = ["S1-3", "S1-1", "S1-2", "S1-4"]
RIGHT = ["S2-10", "S2-2", "S3-7", "S3-1"]


def pairs(rows: list[tuple[str, str]]) -> pa.Table:
    """Build an (s1_id, match_id) pyarrow table."""
    return pa.table(
        {
            "s1_id": pa.array([r[0] for r in rows], pa.string()),
            "match_id": pa.array([r[1] for r in rows], pa.string()),
        }
    )


def s1_table(values: list[str]) -> pa.Table:
    """Build an (s1_id) table."""
    return pa.table({"s1_id": pa.array(values, pa.string())})


def right_table(values: list[str]) -> pa.Table:
    """Build an (entity_id) table of valid right ids."""
    return pa.table({"entity_id": pa.array(values, pa.string())})


CAND = [("S1-1", "S3-7"), ("S1-1", "S2-10"), ("S1-1", "S2-2"), ("S1-2", "S3-1"), ("S1-3", "S2-2")]
MATCH = [("S1-1", "S2-2"), ("S1-1", "S3-7"), ("S1-3", "S2-2")]


def make_test_dir(root: Path) -> Path:
    """Write a tiny synthetic test split for the validator."""
    d = root / "synthetic_test"
    write_text(d / "test_source1.tsv", SOURCE_HEADER + "".join(f"{i}\tn\ta\tUS\n" for i in S1))
    write_text(d / "test_source2.tsv", SOURCE_HEADER + "".join(f"{i}\tn\ta\tUS\n" for i in RIGHT if i[1] == "2"))
    write_text(d / "test_source3.tsv", SOURCE_HEADER + "".join(f"{i}\tn\ta\tUS\n" for i in RIGHT if i[1] == "3"))
    return d


def test_valid_write_passes_official_validator(con, tmp_path: Path) -> None:
    """Deterministic sorted output that the organizer validator (with --check-ids) accepts."""
    out = tmp_path / "output"
    res = write_outputs(con, s1_table(S1), pairs(CAND), pairs(MATCH), right_table(RIGHT), out)
    assert res["n_rows"] == 4 and res["n_empty_match_rows"] == 2 and res["n_empty_candidate_rows"] == 1
    assert (out / "matching_results.tsv").read_bytes().decode() == (
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-2,S3-7\nS1-2\t\nS1-3\tS2-2\nS1-4\t\n"
    )
    assert (out / "candidate_pairs.tsv").read_bytes().decode() == (
        "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-10,S2-2,S3-7\nS1-2\tS3-1\nS1-3\tS2-2\nS1-4\t\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--matching",
            str(out / "matching_results.tsv"),
            "--candidate",
            str(out / "candidate_pairs.tsv"),
            "--test-dir",
            str(make_test_dir(tmp_path)),
            "--check-ids",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
    assert "WARNING" not in proc.stdout.replace("WARNING: ID-existence", "")


@pytest.mark.parametrize(
    "s1, cand, match, message",
    [
        (S1, CAND + [("S1-9", "S2-2")], MATCH, "candidate pair with S1 not in s1_ids"),
        (S1, CAND, MATCH + [("S1-9", "S2-2")], "match pair with S1 not in s1_ids"),
        (S1, CAND + [("S1-2", "S1-3")], MATCH, "candidate right id not an S2-/S3- id"),
        (S1, CAND + [("S1-2", "S2-1,S2-2")], MATCH, "candidate right id not an S2-/S3- id"),
        (S1, CAND + [("S1-2", "S2-999")], MATCH, "candidate right id not in valid_right_ids"),
        (S1, CAND + [("S1-1", "S2-2")], MATCH, "duplicate candidate pair"),
        (S1, CAND, MATCH + [("S1-3", "S2-2")], "duplicate match pair"),
        (S1, CAND, MATCH + [("S1-2", "S2-10")], "match pair not in candidates"),
        (S1 + ["S1-1"], CAND, MATCH, "duplicate s1_id"),
    ],
)
def test_violations_raise(con, tmp_path: Path, s1, cand, match, message) -> None:
    """Each rule violation raises ValueError naming the rule with an example; nothing is written."""
    out = tmp_path / "output"
    with pytest.raises(ValueError, match=message):
        write_outputs(con, s1_table(s1), pairs(cand), pairs(match), right_table(RIGHT), out)
    assert not (out / "matching_results.tsv").exists()
