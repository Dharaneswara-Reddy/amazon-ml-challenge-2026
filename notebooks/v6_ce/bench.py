"""CPU throughput of multilingual-e5-small (MIT, 117.7M) as a pair cross-encoder: inference and training."""
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(10)
name = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
a = ["Lille Club SARL, 24 Rue de Fleurus, Lille, Hauts-de-France"] * 512
b = ["S.A.S Lille Club, 24 R Des Tours, Lille, Nord"] * 512
enc = tok(a, b, truncation=True, max_length=64, padding=True, return_tensors="pt")
print("seq len", enc["input_ids"].shape)
model.eval()
with torch.inference_mode():
    for bs in (128, 256):
        t = time.time()
        for i in range(0, 512, bs):
            model(**{k: v[i:i + bs] for k, v in enc.items()})
        print("inference bs", bs, round(512 / (time.time() - t)), "pairs/s", flush=True)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
y = torch.ones(32)
t = time.time()
for i in range(0, 256, 32):
    out = model(**{k: v[i:i + 32] for k, v in enc.items()})
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits.squeeze(-1), y)
    loss.backward(); opt.step(); opt.zero_grad()
print("train", round(256 / (time.time() - t)), "pairs/s", flush=True)
