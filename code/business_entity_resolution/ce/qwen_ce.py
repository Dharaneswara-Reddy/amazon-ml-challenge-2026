"""LLM pair classifier (D-028): Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0, 1.54B params) + LoRA, fine-tuned only on the
provided train data (bundle train.parquet), run locally on one GPU (no hosted API). Scores score_{holdout,tune,test}.

Setup on the GPU box (Kaggle T4 x2 / Colab T4 / any CUDA GPU):  pip install -U "transformers<5" peft pandas pyarrow
Run:  python qwen_ce.py --bundle ./ber_ce --out ./out [--train-minutes 110] [--max-train 300000]
Writes out/ce_<group>.parquet (lid, rid, ce logit) group by group, so a cut-off session keeps finished groups.
Score only with a saved adapter (no training):  python qwen_ce.py --bundle ./ber_ext --out ./out --lora ./old_out/lora
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Record A: {a}\nRecord B: {b}\nDo A and B describe the same business?"


def main() -> None:
    """Fine-tune the LoRA classifier with a time guard, then score each group of the bundle."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-minutes", type=float, default=110)
    ap.add_argument("--max-train", type=int, default=300_000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=96)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0, help="limit every file to N rows (pipeline check)")
    ap.add_argument("--lora", default="", help="saved adapter dir: skip training and only score the bundle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    dev = args.device
    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.float32 if dev == "cpu" else (torch.bfloat16 if bf16 else torch.float16)  # compute (autocast) dtype
    load_dtype = torch.bfloat16 if bf16 else torch.float32  # fp16 GPUs (T4): fp32 weights + fp16 autocast + GradScaler
    print(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "bf16" if bf16 else "fp16", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1, torch_dtype=load_dtype)
    model.config.pad_token_id = tok.pad_token_id
    if args.lora:
        score_only(args, tok, PeftModel.from_pretrained(model, args.lora).to(dev), dev, dtype)
        return
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora).to(dev)
    for n, p in model.named_parameters():  # train the classification head too (kept in fp32 for stability)
        if "score" in n:
            p.requires_grad = True
            p.data = p.data.float()
    model.print_trainable_parameters()
    # both GPUs (Kaggle T4 x2): each batch is split across devices
    net = torch.nn.DataParallel(model) if dev == "cuda" and torch.cuda.device_count() > 1 else model
    print("gpus", torch.cuda.device_count() if dev == "cuda" else 0, flush=True)

    def enc(a, b):
        """Tokenize prompt-formatted pairs."""
        return tok([PROMPT.format(a=x, b=y) for x, y in zip(a, b)], truncation=True, max_length=args.maxlen,
                   padding=True, return_tensors="pt").to(dev)

    tr = pd.read_parquet(f"{args.bundle}/train.parquet").sample(frac=1.0, random_state=42).reset_index(drop=True)
    if args.smoke:
        tr = tr.iloc[:5000 + args.smoke]
    dv, tr = tr.iloc[:5000].reset_index(drop=True), tr.iloc[5000:5000 + args.max_train].reset_index(drop=True)
    print("train", len(tr), "pos", round(float(tr.label.mean()), 4), flush=True)
    steps = math.ceil(len(tr) / args.bs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.05, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16 and dev == "cuda")
    y = torch.tensor(tr.label.values, dtype=torch.float32)
    model.train()
    t0 = time.time()
    for s in range(steps):
        j = np.arange(s * args.bs, min((s + 1) * args.bs, len(tr)))
        with torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            logit = net(**enc(tr.ta.values[j].tolist(), tr.tb.values[j].tolist())).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y[j].to(dev))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        el = time.time() - t0
        if (s + 1) % 200 == 0:
            print(f"step {s + 1}/{steps} loss {loss.item():.4f} {el / 60:.1f} min, eta {el / (s + 1) * (steps - s - 1) / 60:.1f} min",
                  flush=True)
        if el > args.train_minutes * 60:
            print(f"time guard: stop at step {s + 1}/{steps}", flush=True)
            break

    net.eval()
    score = make_scorer(net, enc, dev, dtype)
    d = score(dv.iloc[:args.smoke] if args.smoke else dv)
    dv = dv.iloc[:args.smoke] if args.smoke else dv
    p = 1 / (1 + np.exp(-d))
    yv = dv.label.values
    ll = -np.mean(yv * np.log(np.clip(p, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-7, 1)))
    print(f"dev logloss {ll:.4f} acc {np.mean((p > 0.5) == yv):.4f}", flush=True)
    model.save_pretrained(f"{args.out}/lora")
    score_groups(args, score)


def make_scorer(net, enc, dev: str, dtype):
    """Return score(df) -> logits for all pairs of df (ta, tb), in length-sorted batches."""

    def score(df: pd.DataFrame, bs: int = 256) -> np.ndarray:
        """Logits for all pairs of df (length-sorted batches)."""
        order = np.argsort((df.ta.str.len() + df.tb.str.len()).values, kind="stable")
        out = np.zeros(len(df), dtype=np.float32)
        t1 = time.time()
        with torch.inference_mode(), torch.autocast(dev, dtype=dtype, enabled=dev == "cuda"):
            for i in range(0, len(df), bs):
                j = order[i:i + bs]
                out[j] = net(**enc(df.ta.values[j].tolist(), df.tb.values[j].tolist())).logits.squeeze(-1).float().cpu().numpy()
                if (i // bs) % 200 == 0:
                    el = time.time() - t1
                    print(f"  {i}/{len(df)} {el / 60:.1f} min, eta {el / (i + bs) * (len(df) - i - bs) / 60:.1f} min", flush=True)
        return out

    return score


def score_groups(args, score) -> None:
    """Score score_{tune,holdout,test}.parquet of the bundle (whichever exist) and write out/ce_<group>.parquet."""
    for g in ("tune", "holdout", "test"):
        path = f"{args.bundle}/score_{g}.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if args.smoke:
            df = df.iloc[:args.smoke].reset_index(drop=True)
        ce = score(df)
        pd.DataFrame({"lid": df.lid, "rid": df.rid, "ce": ce}).to_parquet(f"{args.out}/ce_{g}.parquet", index=False)
        print("wrote", g, len(df), flush=True)
    print("DONE", flush=True)


def score_only(args, tok, model, dev: str, dtype) -> None:
    """Score the bundle with a saved adapter (model already wrapped by PeftModel); no training."""
    model.eval()
    net = torch.nn.DataParallel(model) if dev == "cuda" and torch.cuda.device_count() > 1 else model
    print("score only, gpus", torch.cuda.device_count() if dev == "cuda" else 0, flush=True)

    def enc(a, b):
        """Tokenize prompt-formatted pairs."""
        return tok([PROMPT.format(a=x, b=y) for x, y in zip(a, b)], truncation=True, max_length=args.maxlen,
                   padding=True, return_tensors="pt").to(dev)

    score_groups(args, make_scorer(net, enc, dev, dtype))


if __name__ == "__main__":
    main()
