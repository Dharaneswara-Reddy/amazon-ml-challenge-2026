"""Score a bundle with the fine-tuned multilingual-e5-small cross-encoder (D-028/D-029) on CPU, no training.

Same inputs as ce_colab.ipynb's scoring cell: pair tokenization (ta, tb), max_length 64, logit of the 1-unit head.
Usage: python e5_cpu_score.py MODEL_DIR BUNDLE_DIR OUT_DIR [THREADS]
Reads BUNDLE_DIR/score_{tune,holdout,test}.parquet (whichever exist); writes OUT_DIR/ce_<group>.parquet (lid, rid, ce).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAXLEN, BS = 64, 256


def score(model, tok, df: pd.DataFrame) -> np.ndarray:
    """Logits for all pairs of df, in length-sorted batches; prints progress."""
    order = np.argsort((df.ta.str.len() + df.tb.str.len()).values, kind="stable")
    out = np.zeros(len(df), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(df), BS):
            j = order[i:i + BS]
            enc = tok(df.ta.values[j].tolist(), df.tb.values[j].tolist(), truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt")
            out[j] = model(**enc).logits.squeeze(-1).float().numpy()
            if (i // BS) % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(df)} {el / 60:.1f} min, eta {el / (i + BS) * (len(df) - i - BS) / 60:.1f} min",
                      flush=True)
    return out


def main(model_dir: str, bundle: str, out: str, threads: int) -> None:
    """Score every score_<group>.parquet of the bundle (skipping groups already written) -> ce_<group>.parquet."""
    torch.set_num_threads(threads)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    for g in ("tune", "holdout", "test"):
        path = f"{bundle}/score_{g}.parquet"
        if not os.path.exists(path) or os.path.exists(f"{out}/ce_{g}.parquet"):
            continue
        df = pd.read_parquet(path)
        ce = score(model, tok, df)
        pd.DataFrame({"lid": df.lid, "rid": df.rid, "ce": ce}).to_parquet(f"{out}/ce_{g}.parquet", index=False)
        print("wrote", g, len(df), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else os.cpu_count())
