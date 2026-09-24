"""``ber`` command-line entry point. Each subcommand = (name, add_arguments, run) registered in COMMANDS."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ber.config import DEFAULT_CONFIG, Config, config_hash, load_config
from ber.obs import kv, setup_logging, stage

log = logging.getLogger("ber.cli")

AddArgs = Callable[[argparse.ArgumentParser], None]
Run = Callable[[argparse.Namespace, Config], int]


def _emit(result: dict[str, Any], json_out: Path | None) -> None:
    """Write ``result`` as JSON to stdout and optionally to ``json_out``."""
    text = json.dumps(result, indent=2, default=str)
    sys.stdout.write(text + "\n")
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(text + "\n", encoding="utf-8")


def add_prepare(p: argparse.ArgumentParser) -> None:
    """Arguments for ``ber prepare``."""
    p.add_argument("--force", action="store_true", help="rebuild even if cached parquet is fresh")


def run_prepare(args: argparse.Namespace, cfg: Config) -> int:
    """Convert the organizer TSVs into the Parquet cache and report per-file stats."""
    from ber.io import prepare

    with stage("prepare", cache_dir=cfg.paths.cache_dir):
        results = prepare(cfg, force=args.force)
    _emit(results, None)
    return 0


def add_score(p: argparse.ArgumentParser) -> None:
    """Arguments for ``ber score``."""
    p.add_argument("--pred", type=Path, required=True, help="matching_results.tsv to score")
    p.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="ground truth: GT TSV or pairs parquet (default: cached train_gt_pairs.parquet)",
    )
    p.add_argument(
        "--s1-ids",
        type=Path,
        default=None,
        help="population: parquet with s1_id, or id-list TSV (first column); default: S1 rows of --pred",
    )
    p.add_argument("--cand", type=Path, default=None, help="optional candidate_pairs.tsv for blocking stats")
    p.add_argument(
        "--n-right",
        type=int,
        default=None,
        help="right-side record count for reduction ratio (default: train S2+S3 rows)",
    )
    p.add_argument("--json-out", type=Path, default=None, help="also write the result JSON here")


def run_score(args: argparse.Namespace, cfg: Config) -> int:
    """Score a matching file with macro F0.5 (and optionally blocking stats) against training GT."""
    from ber import io
    from ber.duck import connect
    from ber.evaluate import blocking_stats, macro_f05

    con = connect(cfg)
    try:
        pred, pred_s1 = io.read_id_list_file(con, args.pred)
        gt, _ = io.read_id_list_file(con, args.gt or io.gt_pairs_path(cfg))
        if args.s1_ids is None:
            s1 = pred_s1
        elif args.s1_ids.suffix == ".parquet":
            s1 = con.read_parquet(str(args.s1_ids)).select("s1_id")
        else:
            s1 = io.read_id_list_file(con, args.s1_ids)[1]
        with stage("score", pred=args.pred):
            result: dict[str, Any] = {"pred": str(args.pred), "metrics": macro_f05(con, pred, gt, s1)}
            if args.cand:
                n_right = args.n_right
                if n_right is None:
                    n_right = io.open_right(con, cfg, "train").count("*").fetchone()[0]
                cand, _ = io.read_id_list_file(con, args.cand)
                result["blocking"] = blocking_stats(con, cand, gt, s1, n_right)
    finally:
        con.close()
    log.info(kv(event="score", **result["metrics"]))
    _emit(result, args.json_out)
    return 0


def add_validate_output(p: argparse.ArgumentParser) -> None:
    """Arguments for ``ber validate-output``."""
    p.add_argument(
        "--out-dir", type=Path, default=None, help="dir with both TSVs (default: paths.output_dir)"
    )
    p.add_argument("--test-dir", type=Path, default=None, help="default: <data_dir>/test")
    p.add_argument(
        "--check-ids", action="store_true", help="pass --check-ids (loads all test S2/S3 ids; GBs)"
    )


def validator_command(cfg: Config, out_dir: Path, test_dir: Path, check_ids: bool) -> list[str]:
    """Build the argv that runs the official validator on ``out_dir`` against ``test_dir``."""
    cmd = [
        sys.executable,
        str(cfg.paths.validator),
        "--matching",
        str(out_dir / "matching_results.tsv"),
        "--candidate",
        str(out_dir / "candidate_pairs.tsv"),
        "--test-dir",
        str(test_dir),
    ]
    return cmd + (["--check-ids"] if check_ids else [])


def run_validate_output(args: argparse.Namespace, cfg: Config) -> int:
    """Run the organizer's validate_submission.py from the repo root; return its exit code."""
    out_dir = (args.out_dir or Path(cfg.paths.output_dir)).resolve()
    test_dir = (args.test_dir or Path(cfg.paths.data_dir) / "test").resolve()
    cmd = validator_command(cfg, out_dir, test_dir, args.check_ids)
    log.info(kv(event="validate_output", cmd=" ".join(cmd)))
    return subprocess.run(cmd, cwd=cfg.root, check=False).returncode


COMMANDS: list[tuple[str, str, AddArgs, Run]] = [
    ("prepare", "convert organizer TSVs to the Parquet cache", add_prepare, run_prepare),
    ("score", "macro F0.5 of a matching file against training GT", add_score, run_score),
    ("validate-output", "run the official submission validator", add_validate_output, run_validate_output),
]


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser; common flags are accepted before or after the subcommand."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS, help=f"base config YAML (default: {DEFAULT_CONFIG})"
    )
    common.add_argument(
        "--override", type=Path, default=argparse.SUPPRESS, help="YAML deep-merged over --config"
    )
    common.add_argument("--root", type=Path, default=argparse.SUPPRESS, help="repo root for relative paths")
    common.add_argument("--log-file", type=Path, default=argparse.SUPPRESS, help="also log to this file")
    parser = argparse.ArgumentParser(prog="ber", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text, add_args, run in COMMANDS:
        p = sub.add_parser(name, help=help_text, parents=[common])
        add_args(p)
        p.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, load config, set up logging and dispatch to the subcommand; return an exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "log_file", None))
    cfg = load_config(
        getattr(args, "config", None), getattr(args, "override", None), getattr(args, "root", None)
    )
    log.info(kv(event="cli_start", command=args.command, config_hash=config_hash(cfg), root=cfg.root))
    return int(args.func(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
