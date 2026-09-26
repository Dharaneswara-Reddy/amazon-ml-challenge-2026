"""End-to-end pipeline v2 (docs/DESIGN_V2.md). Every stage reads and writes files under paths.cache_dir or a run dir,
so stages re-run independently and the chain is reproducible from the raw TSVs:

    prepare → learn-maps → normalize → block (sample: dump union) → train-ranker → block → learn-logodds
            → features → train (stage 1 + OOF + stage 2 + decision choice) → predict (+ prior shift) → write

Decision IDs referenced here are recorded in docs/DECISIONS.md and in each run's manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from ber import io
from ber.blocking import BlockingConfig, active_passes, recall_by_pass, register_inputs, run_blocking, train_ranker
from ber.config import Config
from ber.decision import best_per_right, calibrate, choose_decision, decide, em_prior_shift
from ber.duck import connect
from ber.features import FEATURE_SET_VERSION, FEATURES, name_idf, write_features
from ber.manifest import new_run_dir, write_manifest
from ber.model import predict_file, train
from ber.obs import kv, stage
from ber.split import FOLDS, group_where, other_fold
from ber.stage2 import MODEL2_FEATURES, oof_predict, predict_protocol, write_stage2_features
from ber.writer import write_outputs

log = logging.getLogger(__name__)

DECISIONS_IN_EFFECT = [f"D-{i:03d}" for i in range(1, 24)]
TRAIN_GROUPS = ("fit", "tune", "holdout")


# ---------------------------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------------------------


def norm_path(cfg: Config, split: str, source: int) -> Path:
    """Path of the normalized parquet for one split/source."""
    return io.cache_dir(cfg) / f"norm_{split}_s{source}.parquet"


def cand_path(cfg: Config, split: str) -> Path:
    """Path of the blocking candidates parquet for a split."""
    return io.cache_dir(cfg) / f"cand_{split}.parquet"


def model_cand_path(cfg: Config, split: str) -> Path:
    """Candidates the model scores: blocking v3 output (cand3_*) when ``blocking.v3`` is on, else v2 candidates."""
    if bool(cfg.get("blocking", {}).get("v3", False)):
        return io.cache_dir(cfg) / f"cand3_{split}.parquet"
    return cand_path(cfg, split)


def feat_path(cfg: Config, split: str, group: str | None = None) -> Path:
    """Path of a feature parquet (train: one per group incl. rival context; test: one file)."""
    suffix = f"_{group}" if group else ""
    return io.cache_dir(cfg) / f"feat_{FEATURE_SET_VERSION}_{split}{suffix}.parquet"


def learned_dir(cfg: Config) -> Path:
    """Directory of statistics learned from the fit group (maps, ranker, token log-odds)."""
    d = io.cache_dir(cfg) / "learned"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gt_sql(cfg: Config, group: str | None = None) -> str:
    """SQL of train ground-truth pairs (s1_id, match_id), optionally restricted to one entity group."""
    where = f"WHERE {group_where('s1_id', group)}" if group else ""
    return f"SELECT s1_id, match_id FROM read_parquet('{io.gt_pairs_path(cfg)}') {where}"


def blocking_config(cfg: Config, **overrides: Any) -> BlockingConfig:
    """Build the BlockingConfig from the `blocking` config section, the learned ranker if present, and overrides."""
    section = dict(cfg.get("blocking", {}))
    for key in ("v3", "v3_top_k", "v3_threshold"):  # blocking v3 settings, read by block3_stage
        section.pop(key, None)
    for key in ("sn_keys", "token_fields"):
        if key in section:
            section[key] = tuple(section[key])
    ranker = learned_dir(cfg) / "ranker.json"
    if ranker.exists() and "ranker_path" not in overrides:
        section["ranker_path"] = str(ranker)
        section["final_top_k"] = int(json.loads(ranker.read_text())["k"])
    section.update(overrides)
    return BlockingConfig(**section)


def _views(con: duckdb.DuckDBPyConnection, cfg: Config, split: str, r_sample_pct: int | None = None) -> dict:
    """Register views L (Source 1) and R (Sources 2+3) over normalized parquet; optionally sample R by hash."""
    counts = register_inputs(
        con, str(norm_path(cfg, split, 1)), [str(norm_path(cfg, split, 2)), str(norm_path(cfg, split, 3))]
    )
    if r_sample_pct is not None:
        con.execute(
            f"""CREATE OR REPLACE VIEW R AS SELECT * FROM read_parquet(['{norm_path(cfg, split, 2)}',
                  '{norm_path(cfg, split, 3)}']) WHERE md5_number(entity_id) % 100 < {r_sample_pct}"""
        )
        counts["n_right"] = con.execute("SELECT count(*) FROM R").fetchone()[0]
    return counts


# ---------------------------------------------------------------------------------------------------------------
# S1: learned normalization maps + normalize
# ---------------------------------------------------------------------------------------------------------------


def learn_maps_stage(cfg: Config, n_pairs: int = 300_000, n_addr: int = 1_000_000) -> dict[str, Any]:
    """Learn the Indic legal-form token map (D-012) from fit-group true pairs and the admin vocabulary (D-011).

    Only fit-group labels are used; the admin vocabulary uses unlabeled addresses of train and test.
    """
    from ber import normalize as nz

    con = connect(cfg)
    try:
        raw = {(sp, s): io.raw_parquet_path(cfg, sp, s) for sp in ("train", "test") for s in io.SOURCES}
        pairs = con.execute(
            f"""SELECT a.business_name, b.business_name
                FROM ({gt_sql(cfg, "fit")}) g
                JOIN read_parquet('{raw["train", 1]}') a ON a.entity_id = g.s1_id
                JOIN read_parquet(['{raw["train", 2]}', '{raw["train", 3]}']) b ON b.entity_id = g.match_id
                WHERE regexp_matches(b.business_name, '[^\\x00-\\x7F]')
                USING SAMPLE {n_pairs} ROWS (reservoir, 42)"""
        ).fetchall()
        token_map = nz.learn_token_map((nz.normalize_name(a), nz.normalize_name(b)) for a, b in pairs)
        addrs = con.execute(
            f"""SELECT business_address FROM read_parquet([{", ".join(f"'{p}'" for p in raw.values())}])
                USING SAMPLE {n_addr} ROWS (reservoir, 42)"""
        ).fetchall()
        admin = sorted(nz.learn_admin_vocab(nz.normalize_address(a) for (a,) in addrs))
    finally:
        con.close()
    d = learned_dir(cfg)
    (d / "token_map.json").write_text(json.dumps(token_map, indent=1, ensure_ascii=False))
    (d / "admin_vocab.json").write_text(json.dumps(admin, indent=1, ensure_ascii=False))
    log.info(kv(event="learned_maps", token_map_entries=len(token_map), admin_vocab=len(admin), pairs=len(pairs)))
    return {"token_map_entries": len(token_map), "admin_vocab": len(admin), "pairs_used": len(pairs)}


def normalize_stage(cfg: Config, split: str, force: bool = False) -> dict[str, Any]:
    """Normalize the raw parquet of every source of a split (v2 columns), using learned maps when present."""
    from ber.normalize import normalize_parquet

    d = learned_dir(cfg)
    token_map = str(d / "token_map.json") if (d / "token_map.json").exists() else None
    admin = str(d / "admin_vocab.json") if (d / "admin_vocab.json").exists() else None
    out: dict[str, Any] = {"token_map": token_map, "admin_vocab": admin}
    for source in io.SOURCES:
        dst = norm_path(cfg, split, source)
        if dst.exists() and not force:
            out[dst.name] = "cached"
            continue
        with stage("normalize", split=split, source=source) as st:
            normalize_parquet(
                str(io.raw_parquet_path(cfg, split, source)),
                str(dst),
                int(cfg.engine.workers),
                int(cfg.engine.batch_rows),
                token_map_path=token_map,
                admin_vocab_path=admin,
            )
            st["out"] = str(dst)
        out[dst.name] = "written"
    return out


# ---------------------------------------------------------------------------------------------------------------
# S2: blocking (+ ranker training on a sample)
# ---------------------------------------------------------------------------------------------------------------


def block_stage(cfg: Config, split: str, sample_pct: int | None = None) -> dict[str, Any]:
    """Run blocking v2 in a file-backed DuckDB and save candidates + stats (train: recall per pass).

    sample_pct (train only): block a hash sample of right records with legacy ranking and dump the UNCAPPED union
    of fit-group entities for ranker training (D-006, D-018); candidates are then written to a *_sample file.
    """
    sample = sample_pct is not None
    over: dict[str, Any] = {}
    dump = learned_dir(cfg) / "union_sample.parquet"
    if sample:
        over = {"ranker_path": None, "dump_union_path": str(dump), "dump_where": group_where("lid", "fit")}
    bcfg = blocking_config(cfg, **over)
    db = io.cache_dir(cfg) / f"block_{split}{'_sample' if sample else ''}.duckdb"
    db.unlink(missing_ok=True)
    con = connect(cfg, db)
    out_path = cand_path(cfg, split).with_name(f"cand_{split}_sample.parquet") if sample else cand_path(cfg, split)
    try:
        with stage("block", split=split, sample_pct=sample_pct) as st:
            counts = _views(con, cfg, split, sample_pct)
            stats = run_blocking(con, bcfg)
            con.execute(f"COPY candidates TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)")
            per_left = con.execute(
                """SELECT avg(n), quantile_cont(n, 0.5), quantile_cont(n, 0.99), max(n)
                   FROM (SELECT lid, count(*) n FROM candidates GROUP BY lid)"""
            ).fetchone()
            stats.update(counts)
            stats["final_top_k"] = bcfg.final_top_k
            stats["ranker"] = bcfg.ranker_path or "legacy"
            stats["cand_per_left_mean_p50_p99_max"] = per_left
            stats["reduction_ratio"] = 1 - stats["final_pairs"] / (counts["n_left"] * counts["n_right"])
            if split == "train":
                g = gt_sql(cfg)
                if sample:
                    g = f"SELECT * FROM ({g}) WHERE match_id IN (SELECT entity_id FROM R)"
                stats["recall"] = recall_by_pass(con, g, active_passes(bcfg))
            st.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
    finally:
        con.close()
        db.unlink(missing_ok=True)
    return stats


def block3_stage(cfg: Config, split: str) -> dict[str, Any]:
    """Blocking v3 (D-022): compound-key passes over the v2 candidates, union, learned cut -> cand3_<split>.parquet.

    Train: the cut GBDT is trained on FIT-group S1 rows and saved to learned/cut_model.txt; recall is reported overall
    and per group. Test: the saved cut model is applied unchanged.
    """
    from ber import block3

    bsec = dict(cfg.get("blocking", {}))
    top_k, thr = int(bsec.get("v3_top_k", 1)), float(bsec.get("v3_threshold", 1e-3))
    db = io.cache_dir(cfg) / f"block3_{split}.duckdb"
    union_dir = io.cache_dir(cfg) / f"block3_union_{split}"
    db.unlink(missing_ok=True)
    con = connect(cfg, db)
    out = model_cand_path(cfg, split)
    model = learned_dir(cfg) / "cut_model.txt"
    stats: dict[str, Any] = {}
    try:
        with stage("block3", split=split) as st:
            gt = str(io.gt_pairs_path(cfg)) if split == "train" else None
            stats.update(block3.load_tables(
                con, str(norm_path(cfg, split, 1)), [str(norm_path(cfg, split, s)) for s in (2, 3)],
                str(cand_path(cfg, split)), gt,
            ))
            stats.update(block3.run_passes(con, block3.PASSES_V3))
            ranker = learned_dir(cfg) / "ranker.json"
            stats["union_rows"] = block3.build_union(con, union_dir, str(ranker) if ranker.exists() else None)
            for p in block3.PASSES_V3:
                con.execute(f"DROP TABLE IF EXISTS cand_{p}")
            if split == "train":
                stats.update(block3.train_cut(con, union_dir, "bkt < 30", model, int(cfg.seed)))
            stats.update(block3.apply_cut(con, union_dir, model, out, top_k, thr))
            if split == "train":
                stats["recall"] = block3.recall(con, out, gt)
                for g in TRAIN_GROUPS:
                    stats[f"recall_{g}"] = block3.recall(con, out, gt, group_where("s1_id", g))["recall"]
            else:
                nl = con.execute("SELECT count(*) FROM L").fetchone()[0]
                stats["pairs_per_s1"] = stats["final_pairs"] / max(nl, 1)
            st.update({k: v for k, v in stats.items() if not isinstance(v, dict)})
    finally:
        con.close()
        db.unlink(missing_ok=True)
        import shutil

        shutil.rmtree(union_dir, ignore_errors=True)
    return stats


def train_ranker_stage(cfg: Config) -> dict[str, Any]:
    """Fit the linear blocking ranker and cap K on the dumped fit-group union sample (run block --sample first)."""
    con = connect(cfg)
    try:
        dump = learned_dir(cfg) / "union_sample.parquet"
        return train_ranker(con, str(dump), gt_sql(cfg, "fit"), str(learned_dir(cfg) / "ranker.json"))
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------------------------
# S3: learned token statistics + features
# ---------------------------------------------------------------------------------------------------------------


def learn_sigvocab(rows: list[tuple], workers: int) -> dict:
    """Signature edit-token vocabulary (D-021) from labelled (l_name, l_addr, r_name, r_addr, label) pairs.

    Edit operations are extracted in parallel; token counts are merged and turned into smoothed log-odds.
    """
    import math
    import multiprocessing as mp
    from collections import Counter

    from ber import signature

    chunks = [rows[i : i + 50_000] for i in range(0, len(rows), 50_000)]
    with mp.get_context("spawn").Pool(max(1, workers)) as pool:
        parts = pool.map(_count_chunk, chunks)
    neg: dict[str, Counter] = {}
    pos: dict[str, Counter] = {}
    n_pos = n_neg = 0
    for cn, cp, a, b in parts:
        n_neg += a
        n_pos += b
        for k in cn:
            neg.setdefault(k, Counter()).update(cn[k])
            pos.setdefault(k, Counter()).update(cp[k])
    prior = math.log(max(n_pos, 1) / max(n_neg, 1))
    return {
        k: {
            w: round(math.log((neg[k][w] + 1) / (pos[k][w] + 1)) + prior, 3)
            for w in set(neg[k]) | set(pos[k])
            if neg[k][w] + pos[k][w] >= signature.VOCAB_MIN_COUNT
        }
        for k in neg
    }


def _count_chunk(rows: list[tuple]) -> tuple[dict, dict, int, int]:
    """Edit-token counts of one chunk: (non-match counts, match counts, n_non_match, n_match)."""
    from collections import Counter

    from ber import signature

    keys = ("ins", "del", "repl", "rtgt", "ains", "adel")
    cn = {k: Counter() for k in keys}
    cp = {k: Counter() for k in keys}
    n_neg = n_pos = 0
    for ln, la, rn, ra, label in rows:
        toks = signature.edit_tokens(signature.pair_ops(ln or "", la or "", rn or "", ra or ""))
        tgt = cp if label else cn
        if label:
            n_pos += 1
        else:
            n_neg += 1
        for k, ws in toks.items():
            tgt[k].update(ws)
    return cn, cp, n_neg, n_pos


def learn_logodds_stage(cfg: Config, max_pairs: int = 3_000_000, max_sig_pairs: int = 1_000_000) -> dict[str, Any]:
    """Learn token log-odds (D-013) and signature vocabularies (D-021) per stacking fold and for the full fit group.

    Both come from labelled train candidates of the model candidate set (cand3 when blocking v3 is on).
    """
    from ber import textdiff

    con = connect(cfg)
    out: dict[str, Any] = {}
    try:
        _views(con, cfg, "train")
        for name, groups in (("fitA", ("fitA",)), ("fitB", ("fitB",)), ("fit", ("fit",))):
            rows = con.execute(
                f"""SELECT l.name_core, r.name_core, (g.s1_id IS NOT NULL)::INT
                    FROM read_parquet('{model_cand_path(cfg, "train")}') c
                    JOIN L l ON l.entity_id = c.lid JOIN R r ON r.entity_id = c.rid
                    LEFT JOIN ({gt_sql(cfg)}) g ON g.s1_id = c.lid AND g.match_id = c.rid
                    WHERE {group_where("c.lid", *groups)}
                    USING SAMPLE {max_pairs} ROWS (reservoir, 42)"""
            ).fetchall()
            table = textdiff.learn_token_logodds(rows)
            (learned_dir(cfg) / f"logodds_{name}.json").write_text(json.dumps(table, ensure_ascii=False))
            del rows
            sig_rows = con.execute(
                f"""SELECT l.name_norm, l.addr_norm, r.name_norm, r.addr_norm, (g.s1_id IS NOT NULL)::INT
                    FROM read_parquet('{model_cand_path(cfg, "train")}') c
                    JOIN L l ON l.entity_id = c.lid JOIN R r ON r.entity_id = c.rid
                    LEFT JOIN ({gt_sql(cfg)}) g ON g.s1_id = c.lid AND g.match_id = c.rid
                    WHERE {group_where("c.lid", *groups)}
                    USING SAMPLE {max_sig_pairs} ROWS (reservoir, 42)"""
            ).fetchall()
            vocab = learn_sigvocab(sig_rows, int(cfg.engine.workers))
            (learned_dir(cfg) / f"sigvocab_{name}.json").write_text(json.dumps(vocab, ensure_ascii=False))
            out[name] = {"tokens": len(table), "sig_pairs": len(sig_rows), "sig_tokens": sum(map(len, vocab.values()))}
    finally:
        con.close()
    log.info(kv(event="learned_logodds", **{k: v["tokens"] for k, v in out.items()}))
    return out


def _sigvocab(cfg: Config, name: str) -> dict | None:
    """Load a learned signature vocabulary, or None when it has not been learned yet."""
    p = learned_dir(cfg) / f"sigvocab_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _logodds(cfg: Config, name: str) -> dict[str, float] | None:
    """Load a learned log-odds table, or None when it has not been learned yet."""
    p = learned_dir(cfg) / f"logodds_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def features_stage(cfg: Config, split: str) -> dict[str, Any]:
    """Feature parquet(s) v2. Train: fitA/fitB (each with the OTHER fold's log-odds) concatenated into "fit", plus
    tune and holdout (full-fit log-odds); every group file includes rival-context rows (in_group = 0). Test: one file.
    """
    db = io.cache_dir(cfg) / f"features_{split}.duckdb"
    db.unlink(missing_ok=True)
    con = connect(cfg, db)
    workers, rows = int(cfg.engine.workers), int(cfg.engine.batch_rows)
    buckets = int(cfg.get("features", {}).get("buckets", 16))
    out: dict[str, Any] = {}
    try:
        _views(con, cfg, split)
        con.execute(f"CREATE OR REPLACE VIEW candidates AS SELECT * FROM read_parquet('{model_cand_path(cfg, split)}')")
        idf = name_idf(con)
        if split == "train":
            label = gt_sql(cfg)
            parts = []
            for fold in FOLDS:
                p = feat_path(cfg, split, fold)
                with stage("features", split=split, group=fold) as st:
                    st["rows"] = write_features(
                        con,
                        p,
                        label,
                        rows,
                        workers,
                        group_where("c.lid", fold),
                        True,
                        buckets,
                        idf,
                        _logodds(cfg, other_fold(fold)),
                        _sigvocab(cfg, other_fold(fold)),
                    )
                parts.append(p)
                out[fold] = st["rows"]
            files = ", ".join(f"'{p}'" for p in parts)
            # A rival pair can sit in both fold files (and be in-group in one of them): keep one row per pair,
            # preferring the in-group row, whose log-odds features are the correct out-of-fold ones.
            con.execute(
                f"""COPY (SELECT * FROM read_parquet([{files}])
                          QUALIFY row_number() OVER (PARTITION BY lid, rid ORDER BY in_group DESC) = 1)
                    TO '{feat_path(cfg, split, 'fit')}' (FORMAT parquet, COMPRESSION zstd)"""
            )
            for g in ("tune", "holdout"):
                with stage("features", split=split, group=g) as st:
                    st["rows"] = write_features(
                        con,
                        feat_path(cfg, split, g),
                        label,
                        rows,
                        workers,
                        group_where("c.lid", g),
                        True,
                        buckets,
                        idf,
                        _logodds(cfg, "fit"),
                        _sigvocab(cfg, "fit"),
                    )
                out[g] = st["rows"]
        else:
            with stage("features", split=split) as st:
                st["rows"] = write_features(
                    con, feat_path(cfg, split), None, rows, workers, None, False, buckets, idf, _logodds(cfg, "fit"),
                    _sigvocab(cfg, "fit"),
                )
            out["test"] = st["rows"]
        return out
    finally:
        con.close()
        db.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------------
# S4-S6: stage 1, stage 2, decision
# ---------------------------------------------------------------------------------------------------------------


def _in_group_copy(con: duckdb.DuckDBPyConnection, src: Path, dst: Path) -> Path:
    """Copy only in_group = 1 rows (never train on rival-context rows) and return the new path."""
    con.execute(f"COPY (SELECT * FROM read_parquet('{src}') WHERE in_group = 1) TO '{dst}' (FORMAT parquet)")
    return dst


def _fetch_numpy(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, np.ndarray]:
    """Run ``sql`` into a temp table, then fetch the table as numpy arrays on one thread.

    Fetching a large join/window query directly streams it through DuckDB's buffered result path; with 12 threads
    that livelocked (one thread spinning on a mutex for 35+ minutes on 4M rows, DuckDB 1.5.5) while the same query
    took 7 s in a fresh connection. Materializing runs the query fully in parallel; the plain scan is then fetched
    with threads = 1 so no producer threads compete for the result buffer.
    """
    con.execute(f"CREATE OR REPLACE TEMP TABLE _fetch AS {sql}")
    threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
    con.execute("SET threads = 1")
    try:
        return con.execute("SELECT * FROM _fetch").fetchnumpy()
    finally:
        con.execute(f"SET threads = {int(threads)}")
        con.execute("DROP TABLE IF EXISTS _fetch")


def _group_arrays(con: duckdb.DuckDBPyConnection, cfg: Config, preds: Path, group: str) -> dict[str, np.ndarray]:
    """Arrays for the decision layer on one train group, with rivals respected (D-003).

    Every entity of the group is present (n_true, even with no candidates). Pairs of right records whose best
    candidate over ALL rows (incl. rival-context rows) belongs to an entity outside the group are dropped, since
    one-to-one assignment would give them to the rival.
    """
    s1 = str(io.raw_parquet_path(cfg, "train", 1))
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE s1g AS
            SELECT s.entity_id AS s1_id, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
                   coalesce(g.n_matches, 0)::DOUBLE AS n_true
            FROM read_parquet('{s1}') s
            LEFT JOIN read_parquet('{io.gt_s1_path(cfg)}') g ON g.s1_id = s.entity_id
            WHERE {group_where("s.entity_id", group)}"""
    )
    n_true = _fetch_numpy(con, "SELECT n_true FROM s1g ORDER BY idx")["n_true"]
    pairs = _fetch_numpy(
        con,
        f"""WITH pr AS (SELECT lid, rid, p, label FROM read_parquet('{preds}')),
                 best AS (SELECT rid, arg_max(lid, p) AS best_lid FROM pr GROUP BY rid)
            SELECT s.idx, dense_rank() OVER (ORDER BY pr.rid) - 1 AS rc, pr.p, pr.label::BOOLEAN AS label
            FROM pr JOIN s1g s ON s.s1_id = pr.lid JOIN best b ON b.rid = pr.rid
            WHERE b.best_lid IN (SELECT s1_id FROM s1g)""",
    )
    return {
        "lid_idx": pairs["idx"].astype(np.int64),
        "rid": pairs["rc"].astype(np.int64),
        "p": pairs["p"].astype(np.float64),
        "label": pairs["label"].astype(bool),
        "n_true": n_true,
    }


def _complete(path: Path) -> bool:
    """True if ``path`` is a finished parquet file (its footer is written; a killed writer leaves none)."""
    try:
        pq.ParquetFile(path).metadata
        return True
    except Exception:
        return False


def _best_iter(booster: lgb.Booster) -> int:
    """Best iteration of a freshly trained booster, or the tree count of one loaded from a saved (best-only) model."""
    return int(booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration())


def train_stage(cfg: Config, name: str = "v2", resume: Path | None = None) -> Path:
    """Stage-1 LightGBM, out-of-fold stage-1 predictions, stage-2 LightGBM, and the decision rule; score holdout.

    The rule (threshold vs expected-F0.5) and its parameters come from the tune group only (choose_decision);
    the holdout is scored once. Everything is recorded in the run manifest.
    resume: an earlier run dir of this function; every model and every complete prediction / feature parquet
    already there is reused, and only the missing steps run.
    """
    run = Path(resume) if resume else new_run_dir(cfg, name)

    def have(fname: str) -> bool:
        """True when resuming and ``fname`` already exists in the run dir."""
        if resume is None or not (run / fname).exists():
            return False
        return fname.endswith(".txt") or _complete(run / fname)

    seed, threads, rows = int(cfg.seed), int(cfg.engine.duckdb_threads), int(cfg.engine.batch_rows)
    mcfg = dict(cfg.get("model", {}))
    params, rounds = dict(mcfg.get("params", {})), int(mcfg.get("rounds", 5000))
    feat = {g: feat_path(cfg, "train", g) for g in TRAIN_GROUPS}
    con = connect(cfg, run / "work.duckdb")
    metrics: dict[str, Any] = {}
    try:
        fit_in, tune_in = run / "fit_in.parquet", run / "tune_in.parquet"
        for src, dst in ((feat["fit"], fit_in), (feat["tune"], tune_in)):
            if not have(dst.name):
                _in_group_copy(con, src, dst)
        if have("model1.txt"):
            b1 = lgb.Booster(model_file=str(run / "model1.txt"))
        else:
            with stage("train_stage1", run=run) as st:
                b1 = train(fit_in, tune_in, params, seed, threads, rounds=rounds)
                b1.save_model(str(run / "model1.txt"), num_iteration=b1.best_iteration)
                st["best_iteration"] = b1.best_iteration
        if not have("pred1_fit.parquet"):
            with stage("oof_stage1"):
                folds = oof_predict(
                    con, fit_in, tune_in, run / "pred1_fit_in.parquet", run, params, seed, threads, rows
                )
                predict_protocol(con, feat["fit"], run / "pred1_fit.parquet", folds, b1, rows, run)
        for g in ("tune", "holdout"):
            if not have(f"pred1_{g}.parquet"):
                predict_file(b1, feat[g], run / f"pred1_{g}.parquet", rows)
        for g in TRAIN_GROUPS:
            if not have(f"s2_{g}.parquet"):
                write_stage2_features(con, feat[g], run / f"pred1_{g}.parquet", run / f"s2_{g}.parquet")
        s2_fit_in, s2_tune_in = run / "s2_fit_in.parquet", run / "s2_tune_in.parquet"
        for src, dst in ((run / "s2_fit.parquet", s2_fit_in), (run / "s2_tune.parquet", s2_tune_in)):
            if not have(dst.name):
                _in_group_copy(con, src, dst)
        if have("model2.txt"):
            b2 = lgb.Booster(model_file=str(run / "model2.txt"))
        else:
            with stage("train_stage2", run=run) as st:
                b2 = train(s2_fit_in, s2_tune_in, params, seed, threads, rounds=rounds, features=MODEL2_FEATURES)
                b2.save_model(str(run / "model2.txt"), num_iteration=b2.best_iteration)
                st["best_iteration"] = b2.best_iteration
        for g in ("tune", "holdout"):
            if not have(f"pred2_{g}.parquet"):
                predict_file(
                    b2, run / f"s2_{g}.parquet", run / f"pred2_{g}.parquet", rows, features=MODEL2_FEATURES
                )
        log.info(kv(event="decision_arrays_start", run=run))
        arr1 = {g: _group_arrays(con, cfg, run / f"pred1_{g}.parquet", g) for g in ("tune", "holdout")}
        arr2 = {g: _group_arrays(con, cfg, run / f"pred2_{g}.parquet", g) for g in ("tune", "holdout")}
        rep1 = choose_decision(arr1["tune"], arr1["holdout"])
        rep2 = choose_decision(arr2["tune"], arr2["holdout"])
        rule = rep2["params"]["rule"]
        t = arr2["tune"]
        metrics = {
            "stage1_holdout": rep1["holdout"],
            "stage2_holdout": rep2["holdout"],
            "stage2_tune_half_b": rep2["selection"]["tune_half_b"],
            "rule": rule,
            "holdout_f05": rep2["holdout"][rule],
            "holdout_candidate_recall": float(arr2["holdout"]["label"].sum() / max(arr2["holdout"]["n_true"].sum(), 1)),
            "train_prior_best": float(t["label"][best_per_right(t["rid"], t["lid_idx"], t["p"])].mean()),
            "best_iteration_stage1": _best_iter(b1),
            "best_iteration_stage2": _best_iter(b2),
        }
        (run / "decision_params.json").write_text(json.dumps(rep2["params"], default=float))
    finally:
        con.close()
        (run / "work.duckdb").unlink(missing_ok=True)
    importance = dict(zip(MODEL2_FEATURES, b2.feature_importance("gain").round(1).tolist(), strict=True))
    write_manifest(
        run,
        cfg,
        metrics=metrics,
        feature_set=FEATURE_SET_VERSION,
        decisions=DECISIONS_IN_EFFECT,
        importance_gain_stage2=importance,
        n_features_stage1=len(FEATURES),
    )
    _append_experiment(cfg, run, metrics)
    log.info(kv(event="train_done", run_dir=run, rule=rule, holdout_f05=metrics["holdout_f05"]))
    return run


def _append_experiment(cfg: Config, run: Path, metrics: dict[str, Any]) -> None:
    """Append one row to docs/EXPERIMENTS.md (D-020)."""
    path = Path(cfg.root) / "docs" / "EXPERIMENTS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiments\n\n| Run | Rule | Holdout F0.5 | Stage-1 holdout (thr / expF) | Candidate recall | Notes |\n"
            "|---|---|---|---|---|---|\n"
        )
    s1 = metrics["stage1_holdout"]
    row = (
        f"| `{run.name}` | {metrics['rule']} | {metrics['holdout_f05']:.4f} | "
        f"{s1['threshold']:.4f} / {s1['expected_f']:.4f} | {metrics['holdout_candidate_recall']:.4f} | |\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row)


def predict_stage(cfg: Config, run_dir: Path, prior_shift: bool = True, resume: bool = False) -> dict[str, Any]:
    """Score test: stage 1 → stage-2 features → stage 2 → prior-shift correction (D-010) → decision → write outputs.

    resume: reuse complete pred1_test / s2_test / pred2_test parquets already in ``run_dir``. The decision works on
    integer entity / right-record indices only; ids stay in DuckDB and are joined back for the selected pairs
    (holding 30M id strings as Python objects exceeded the 15 GB laptop and was OOM-killed).
    """
    params = json.loads((run_dir / "decision_params.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    b1 = lgb.Booster(model_file=str(run_dir / "model1.txt"))
    b2 = lgb.Booster(model_file=str(run_dir / "model2.txt"))
    rows = int(cfg.engine.batch_rows)
    con = connect(cfg, run_dir / "work_test.duckdb")
    try:
        def done(fname: str) -> bool:
            """True when resuming and ``fname`` is a complete parquet in the run dir."""
            return resume and _complete(run_dir / fname)

        with stage("predict", run=run_dir):
            if not done("pred1_test.parquet"):
                predict_file(b1, feat_path(cfg, "test"), run_dir / "pred1_test.parquet", rows)
            if not done("s2_test.parquet"):
                write_stage2_features(
                    con, feat_path(cfg, "test"), run_dir / "pred1_test.parquet", run_dir / "s2_test.parquet"
                )
            if not done("pred2_test.parquet"):
                predict_file(
                    b2, run_dir / "s2_test.parquet", run_dir / "pred2_test.parquet", rows, features=MODEL2_FEATURES
                )
        s1_raw = io.raw_parquet_path(cfg, "test", 1)
        con.execute(
            f"""CREATE OR REPLACE TEMP TABLE pt AS
                SELECT row_number() OVER () - 1 AS i, x.lid, x.rid, x.p,
                       dense_rank() OVER (ORDER BY x.lid) - 1 AS li, dense_rank() OVER (ORDER BY x.rid) - 1 AS rc,
                       coalesce(s.country, '') AS country
                FROM read_parquet('{run_dir / "pred2_test.parquet"}') x
                LEFT JOIN read_parquet('{s1_raw}') s ON s.entity_id = x.lid"""
        )
        countries = [r[0] for r in con.execute("SELECT DISTINCT country FROM pt ORDER BY 1").fetchall()]
        d = _fetch_numpy(
            con, "SELECT li, rc, p, (dense_rank() OVER (ORDER BY country) - 1)::INT AS cc FROM pt ORDER BY i"
        )
        li, rc, p = d["li"].astype(np.int64), d["rc"].astype(np.int64), d["p"].astype(np.float64)
        cc = d["cc"].astype(np.int64)
        del d
        info: dict[str, Any] = {"rule": params["rule"]}
        pair_shift = None
        if prior_shift:
            # D-023: EM prior shift estimated per country (France has no training prior; pooled shift over-corrects it)
            per_country = bool(cfg.get("decision", {}).get("per_country_shift", True))
            is_best = best_per_right(rc, li, p)
            p_cal = calibrate(params, is_best, p, shift=0.0)
            train_prior = float(manifest["metrics"]["train_prior_best"])
            base = float(params.get("shift", 0.0))
            pair_shift = np.full(len(p), base)
            logit = lambda q: float(np.log(q / (1 - q)))  # noqa: E731
            groups = list(enumerate(countries)) if per_country else [(-1, "all")]
            shifts: dict[str, Any] = {}
            for code, name in groups:
                m = np.ones(len(p), dtype=bool) if code < 0 else cc == code
                test_prior, _ = em_prior_shift(p_cal[is_best & m], train_prior)
                delta = logit(test_prior) - logit(train_prior)
                pair_shift[m] += delta
                shifts[name] = {"test_prior_em": test_prior, "logit_delta": delta, "pairs": int(m.sum())}
            info.update(train_prior=train_prior, prior_shift_by_country=shifts)
            log.info(kv(event="prior_shift", train_prior=train_prior, shifts=json.dumps(shifts, default=float)))
        if params["rule"] == "threshold" and pair_shift is not None:
            is_best = best_per_right(rc, li, p)
            p_cal = calibrate(params, is_best, p, shift=pair_shift)
            t_cal = float(calibrate(params, np.array([True]), np.array([params["threshold"]]), shift=0.0)[0])
            mask = is_best & (p_cal >= t_cal)
        else:
            mask = decide(params, li, rc, p, pair_shift)
        con.register("sel", {"i": np.flatnonzero(mask).astype(np.int64)})
        con.execute("CREATE TEMP TABLE matches AS SELECT pt.lid AS s1_id, pt.rid AS match_id FROM pt JOIN sel USING (i)")
        con.execute("DROP TABLE pt")
        _output_views(con, cfg)
        with stage("write_outputs") as st:
            result = write_outputs(con, "s1", "cands", "matches", "valid_right", Path(cfg.paths.output_dir))
            st.update({k: v for k, v in result.items() if not isinstance(v, dict)})
        info["matches_per_entity_by_country"] = con.execute(
            f"""SELECT l.country, count(DISTINCT l.entity_id) n,
                       count(m.s1_id)::DOUBLE / count(DISTINCT l.entity_id) AS matches_per_entity
                FROM read_parquet('{io.raw_parquet_path(cfg, "test", 1)}') l
                LEFT JOIN matches m ON m.s1_id = l.entity_id GROUP BY 1 ORDER BY 2 DESC"""
        ).fetchall()
    finally:
        con.close()
        (run_dir / "work_test.duckdb").unlink(missing_ok=True)
    (run_dir / "predict.json").write_text(json.dumps({**info, **result}, indent=2, default=str))
    return {**info, **result}


def _output_views(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Views (streamed from parquet, never materialized) for the writer: test S1 ids, candidates, valid right ids."""
    con.execute(
        f"""CREATE OR REPLACE VIEW cands AS
            SELECT lid AS s1_id, rid AS match_id FROM read_parquet('{model_cand_path(cfg, "test")}')"""
    )
    s1_path = io.raw_parquet_path(cfg, "test", 1)
    con.execute(f"CREATE OR REPLACE VIEW s1 AS SELECT entity_id AS s1_id FROM read_parquet('{s1_path}')")
    r2, r3 = io.raw_parquet_path(cfg, "test", 2), io.raw_parquet_path(cfg, "test", 3)
    con.execute(f"CREATE OR REPLACE VIEW valid_right AS SELECT entity_id FROM read_parquet(['{r2}', '{r3}'])")
