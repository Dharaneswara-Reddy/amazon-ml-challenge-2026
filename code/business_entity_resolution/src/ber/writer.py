"""Write the two submission TSVs (matching_results.tsv, candidate_pairs.tsv) with strict pre-checks.

Input convention as in ``ber.evaluate``: ``s1_ids`` has column ``s1_id``; ``cand_pairs`` and
``match_pairs`` have ``s1_id``, ``match_id``; ``valid_right_ids`` has column ``entity_id`` (all S2/S3
ids of the split). Output: one row per S1 id, rows sorted by id, ids sorted inside each list,
comma-joined without spaces, empty string for none, TAB separated, header, no quoting, ``\\n`` endings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import duckdb

from ber.duck import TableLike, as_view, drop_views
from ber.io import count_data_rows, read_header, sql_str
from ber.obs import kv

log = logging.getLogger("ber.writer")

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
RIGHT_ID_RE = "S[23]-[^,\\s]+"
MAX_EXAMPLES = 5


def _violations(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[int, list[str]]:
    """Run ``sql`` (one text column of offenders) and return (count, first sorted examples)."""
    n = con.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    if not n:
        return 0, []
    ex = con.execute(f"SELECT * FROM ({sql}) ORDER BY 1 LIMIT {MAX_EXAMPLES}").fetchall()
    return int(n), [str(r[0]) for r in ex]


def check_inputs(con: duckdb.DuckDBPyConnection, s1: str, cand: str, match: str, valid: str | None) -> None:
    """Raise ValueError listing every rule violated by the input views (counts + examples).

    Rules: s1_ids unique, non-null and TAB/newline/comma free; every pair's S1 in s1_ids; right ids
    match ``S2-``/``S3-`` with no comma/whitespace; right ids exist in ``valid`` (when given);
    no duplicate pairs; every match pair present in candidates.
    """
    checks: list[tuple[str, str]] = [
        (
            "null or malformed s1_id",
            f"SELECT coalesce(s1_id, '<NULL>') FROM {s1} "
            f"WHERE s1_id IS NULL OR s1_id = '' OR regexp_matches(s1_id, '[\\t\\n\\r,]')",
        ),
        ("duplicate s1_id", f"SELECT s1_id FROM {s1} GROUP BY s1_id HAVING count(*) > 1"),
    ]
    for label, view in (("candidate", cand), ("match", match)):
        pair_txt = "s1_id || ' ' || coalesce(match_id, '<NULL>')"
        checks += [
            (
                f"{label} pair with S1 not in s1_ids",
                f"SELECT DISTINCT coalesce(s1_id, '<NULL>') FROM {view} "
                f"WHERE s1_id IS NULL OR s1_id NOT IN (SELECT s1_id FROM {s1})",
            ),
            (
                f"{label} right id not an S2-/S3- id",
                f"SELECT DISTINCT coalesce(match_id, '<NULL>') FROM {view} "
                f"WHERE match_id IS NULL OR NOT regexp_full_match(match_id, '{RIGHT_ID_RE}')",
            ),
            (
                f"duplicate {label} pair",
                f"SELECT {pair_txt} FROM {view} GROUP BY s1_id, match_id HAVING count(*) > 1",
            ),
        ]
        if valid is not None:
            checks.append(
                (
                    f"{label} right id not in valid_right_ids",
                    f"SELECT DISTINCT match_id FROM {view} WHERE match_id IS NOT NULL "
                    f"AND match_id NOT IN (SELECT entity_id FROM {valid})",
                )
            )
    checks.append(
        (
            "match pair not in candidates",
            f"SELECT m.s1_id || ' ' || m.match_id FROM {match} m ANTI JOIN {cand} c USING (s1_id, match_id)",
        )
    )
    problems = []
    for label, sql in checks:
        n, ex = _violations(con, sql)
        if n:
            problems.append(f"{label}: {n} (e.g. {', '.join(ex)})")
    if problems:
        raise ValueError("write_outputs input check failed:\n  " + "\n  ".join(problems))


def _write_list_file(con: duckdb.DuckDBPyConnection, s1: str, pairs: str, header: tuple[str, str], path: Path) -> None:
    """COPY one id-list TSV atomically: one row per S1 id, sorted rows and sorted comma-joined ids."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT s.s1_id AS {header[0]},
                   NULLIF(coalesce(string_agg(p.match_id, ',' ORDER BY p.match_id), ''), '') AS {header[1]}
            FROM (SELECT DISTINCT s1_id FROM {s1}) s LEFT JOIN {pairs} p USING (s1_id)
            GROUP BY s.s1_id
            ORDER BY s.s1_id
        ) TO {sql_str(tmp)} (FORMAT csv, DELIMITER '\t', HEADER true, QUOTE '', ESCAPE '', NEW_LINE '\n')
        """
    )
    os.replace(tmp, path)


def _verify_file(path: Path, header: tuple[str, str], expected_rows: int) -> None:
    """Re-read ``path``: header must equal ``header`` and data row count must equal ``expected_rows``."""
    got_header = tuple(read_header(path))
    if got_header != header:
        raise AssertionError(f"{path}: header {got_header} != {header}")
    rows = count_data_rows(path)
    if rows != expected_rows:
        raise AssertionError(f"{path}: wrote {rows} rows, expected {expected_rows}")
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            if b"\r" in chunk or b'"' in chunk or b" " in chunk:
                raise AssertionError(f"{path}: unexpected CR, quote or space characters")


def write_outputs(
    con: duckdb.DuckDBPyConnection,
    s1_ids: TableLike,
    cand_pairs: TableLike,
    match_pairs: TableLike,
    valid_right_ids: TableLike | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Validate inputs, write both submission TSVs into ``out_dir`` and verify them.

    ``valid_right_ids=None`` skips only the existence check (logged). Raises ValueError on any input
    violation (nothing is written). Returns paths, row count, pair counts and empty-row counts.
    """
    views = [
        as_view(con, s1_ids, ("s1_id",)),
        as_view(con, cand_pairs, ("s1_id", "match_id")),
        as_view(con, match_pairs, ("s1_id", "match_id")),
    ]
    if valid_right_ids is not None:
        views.append(as_view(con, valid_right_ids, ("entity_id",)))
    else:
        log.warning(kv(event="writer_skip_check", check="valid_right_ids"))
    s1, cand, match = views[:3]
    valid = views[3] if len(views) > 3 else None
    try:
        check_inputs(con, s1, cand, match, valid)
        n_s1, n_cand, n_match, empty_cand, empty_match = con.execute(
            f"""SELECT (SELECT count(*) FROM {s1}), (SELECT count(*) FROM {cand}),
                (SELECT count(*) FROM {match}),
                (SELECT count(*) FROM {s1} WHERE s1_id NOT IN (SELECT s1_id FROM {cand})),
                (SELECT count(*) FROM {s1} WHERE s1_id NOT IN (SELECT s1_id FROM {match}))"""
        ).fetchone()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        match_path, cand_path = out_dir / MATCHING_FILE, out_dir / CANDIDATE_FILE
        _write_list_file(con, s1, cand, CANDIDATE_HEADER, cand_path)
        _write_list_file(con, s1, match, MATCHING_HEADER, match_path)
    finally:
        drop_views(con, *views)
    _verify_file(cand_path, CANDIDATE_HEADER, n_s1)
    _verify_file(match_path, MATCHING_HEADER, n_s1)
    result = {
        "matching_path": str(match_path),
        "candidate_path": str(cand_path),
        "n_rows": int(n_s1),
        "n_candidate_pairs": int(n_cand),
        "n_match_pairs": int(n_match),
        "n_empty_candidate_rows": int(empty_cand),
        "n_empty_match_rows": int(empty_match),
    }
    log.info(kv(event="write_outputs", **result))
    return result
