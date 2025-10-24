"""
Reward model training using HuggingFace Transformers (pairwise loss).

This script implements a simple pairwise reward model trainer that expects
`datasets_for_training/pairwise_prefs.csv` with columns:
  question, preferred_answer, other_answer, preference

It requires: torch, transformers, datasets.

Behavior:
- If required libraries are missing, exits with code 2 and prints an
  informational message (so CI/tests can assert this behavior).
- If libraries are present, performs a small training loop using
  AutoModelForSequenceClassification (num_labels=1) computing
  pairwise logistic loss: -log(sigmoid(score_pref - score_other)).

This is a general-purpose trainer; tune model name and hyperparams via
command-line args.
"""
import argparse
import csv
import os
import sys
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fail_missing_libs():
    print("Missing required libraries for HF training. Install `torch`, `transformers`, and `datasets`.")
    sys.exit(2)


try:
    import torch
    # require a reasonably recent torch to avoid subtle incompatibilities
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0])
        tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        # treat older torch as missing to force user to install matching version
        fail_missing_libs()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader, Dataset
except Exception:
    # avoid stack trace in normal flows; tests assert this exit
    fail_missing_libs()


class PairwiseDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = r.get("question", "")
        pref = r.get("preferred_answer", "")
        other = r.get("other_answer", "")
        # Prepare two tokenized examples
        a = self.tokenizer(prompt + "\n" + pref, truncation=True, max_length=self.max_length, return_tensors="pt")
        b = self.tokenizer(prompt + "\n" + other, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"a": a, "b": b}


def read_pairs(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("preferred_answer") or not r.get("other_answer"):
                continue
            rows.append(r)
    return rows


def pairwise_loss(score_pref, score_other):
    # uses -log(sigmoid(s_pref - s_other))
    x = score_pref - score_other
    return -torch.log(torch.sigmoid(x) + 1e-12).mean()


def collate_batch(batch):
    # batch is list of dicts with tok tensors inside; we will stack input_ids and attention_mask
    def stack(key):
        return torch.cat([item[key] for item in batch], dim=0)

    # Each item has a and b tokenized dicts with tensors of shape (1, seq_len)
    a_input_ids = torch.cat([item["a"]["input_ids"] for item in batch], dim=0)
    a_attn = torch.cat([item["a"]["attention_mask"] for item in batch], dim=0)
    b_input_ids = torch.cat([item["b"]["input_ids"] for item in batch], dim=0)
    b_attn = torch.cat([item["b"]["attention_mask"] for item in batch], dim=0)
    return {
        "a_input_ids": a_input_ids,
        "a_attention_mask": a_attn,
        "b_input_ids": b_input_ids,
        "b_attention_mask": b_attn,
    }


def train(args):
    data_path = os.path.join(args.root, "datasets_for_training", "pairwise_prefs.csv")
    if not os.path.exists(data_path):
        logging.error("Pairwise dataset not found at %s", data_path)
        return 3

    rows = read_pairs(data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    ds = PairwiseDataset(rows, tokenizer, max_length=args.max_length)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dl:
            a_in = {"input_ids": batch["a_input_ids"].to(device), "attention_mask": batch["a_attention_mask"].to(device)}
            b_in = {"input_ids": batch["b_input_ids"].to(device), "attention_mask": batch["b_attention_mask"].to(device)}
            optim.zero_grad()
            out_a = model(**a_in).logits.squeeze(-1)
            out_b = model(**b_in).logits.squeeze(-1)
            loss = pairwise_loss(out_a, out_b)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        logging.info("Epoch %d loss=%.6f", epoch + 1, total_loss / len(dl))

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "reward_model_hf")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info("Saved reward model to %s", save_path)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA available")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        return train(args)
    except SystemExit as e:
        raise
    except Exception as e:
        logging.exception("Training failed: %s", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
