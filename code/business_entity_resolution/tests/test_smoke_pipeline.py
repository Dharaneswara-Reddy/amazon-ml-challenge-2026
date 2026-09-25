"""End-to-end smoke test: synthetic TSVs → prepare → normalize → block → features → train → predict → validator."""

import random
import shutil
import subprocess
import sys
from pathlib import Path

from ber import io
from ber.config import PACKAGE_DIR, load_config
from ber.pipeline import (
    block_stage,
    features_stage,
    learn_logodds_stage,
    learn_maps_stage,
    normalize_stage,
    predict_stage,
    train_stage,
)

WORDS = [
    "ram",
    "sharma",
    "blue",
    "ridge",
    "pacific",
    "rapid",
    "quality",
    "west",
    "siddhi",
    "king",
    "vision",
    "prime",
    "golden",
    "star",
    "delta",
    "orbit",
    "maple",
    "lotus",
    "river",
    "summit",
    "nova",
    "zenith",
]
KINDS = ["traders", "marketing", "labs", "clinic", "foods", "motors", "textiles", "services"]
SUFFIX = {"US": ["Inc", "LLC", "Corp"], "India": ["Private Limited", "Pvt Ltd", "LLP"], "France": ["SARL", "SAS"]}
CITY = {
    "US": ["Austin, TX", "Tyler, TX", "Goshen, MA"],
    "India": ["Indore, Madhya Pradesh", "Mylapore, Chennai"],
    "France": ["Bordeaux, Nouvelle-Aquitaine", "Lyon, Auvergne-Rhone-Alpes"],
}


def _noisy(rng: random.Random, name: str, addr: str) -> tuple[str, str]:
    """Return a noisy copy of a name/address pair (case, typo, suffix drop, junk prefix)."""
    n, a = name, addr
    if rng.random() < 0.4:
        n = n.upper()
    if rng.random() < 0.4 and len(n) > 6:
        i = rng.randrange(1, len(n) - 2)
        n = n[:i] + n[i + 1] + n[i] + n[i + 2 :]
    if rng.random() < 0.3:
        n = n.rsplit(" ", 1)[0]
    if rng.random() < 0.2:
        n = "-- " + n
    if rng.random() < 0.5:
        a = a.upper().replace("STREET", "ST").replace("ROAD", "RD")
    return n, a


def _world(root: Path, split: str, n_s1: int, countries: list[str], seed: int) -> None:
    """Write synthetic source TSVs (and ground truth for train) under root/dataset/<split>/."""
    rng = random.Random(seed)
    d = root / "dataset" / split
    d.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    nid = iter(range(10_000_000, 99_999_999, 7919))
    for _ in range(n_s1):
        c = rng.choice(countries)
        name = f"{rng.choice(WORDS).title()} {rng.choice(WORDS).title()} {rng.choice(KINDS).title()} " + rng.choice(
            SUFFIX[c]
        )
        addr = f"{rng.randint(1, 9999)} {rng.choice(WORDS).title()} Road, {rng.choice(CITY[c])}"
        sid = f"S1-{next(nid)}"
        s1.append((sid, name, addr, c))
        matches = []
        if rng.random() > 0.06:
            for _ in range(rng.randint(1, 4)):
                src = rng.choice([2, 3])
                rid = f"S{src}-{next(nid)}"
                (s2 if src == 2 else s3).append((rid, *_noisy(rng, name, addr), c))
                matches.append(rid)
        gt.append((sid, ",".join(matches)))
    for _ in range(n_s1 // 3):
        c = rng.choice(countries)
        src = rng.choice([2, 3])
        name = f"{rng.choice(WORDS).title()} {rng.choice(KINDS).title()}"
        (s2 if src == 2 else s3).append((f"S{src}-{next(nid)}", name, f"{rng.randint(1, 999)} Main Street", c))
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for n, rows in ((1, s1), (2, s2), (3, s3)):
        (d / f"{split}_source{n}.tsv").write_text(header + "".join("\t".join(r) + "\n" for r in rows))
    if split == "train":
        gt_text = "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{a}\t{b}\n" for a, b in gt)
        (d / "train_ground_truth.tsv").write_text(gt_text)


def test_pipeline_end_to_end_passes_official_validator(tmp_path: Path) -> None:
    """The full chain runs on synthetic data and its outputs pass the organizer validator."""
    _world(tmp_path, "train", 3000, ["US", "India"], seed=1)
    _world(tmp_path, "test", 600, ["US", "India", "France"], seed=2)
    (tmp_path / "code").mkdir()
    validator = PACKAGE_DIR.parents[1] / "student_resource" / "utils" / "validate_submission.py"
    dst = tmp_path / "student_resource" / "utils"
    dst.mkdir(parents=True)
    shutil.copy(validator, dst / "validate_submission.py")
    cfg = load_config(root=tmp_path)

    io.prepare(cfg)
    learn_maps_stage(cfg, n_pairs=2000, n_addr=5000)
    for split in ("train", "test"):
        normalize_stage(cfg, split)
        stats = block_stage(cfg, split)
    learn_logodds_stage(cfg, max_pairs=20000)
    for split in ("train", "test"):
        features_stage(cfg, split)
    run_dir = train_stage(cfg, "smoke")
    result = predict_stage(cfg, run_dir)

    assert stats["final_pairs"] > 0
    assert (run_dir / "model1.txt").exists() and (run_dir / "model2.txt").exists()
    out = Path(cfg.paths.output_dir)
    assert result is not None
    proc = subprocess.run(
        [
            sys.executable,
            str(dst / "validate_submission.py"),
            "--matching",
            str(out / "matching_results.tsv"),
            "--candidate",
            str(out / "candidate_pairs.tsv"),
            "--test-dir",
            str(tmp_path / "dataset" / "test"),
            "--check-ids",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
