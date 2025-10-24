"""
Production-ready reward model trainer (pairwise) using HuggingFace/Transformers.

Features:
- YAML/CLI configuration
- logging with rotating file handler
- checkpointing per epoch and best-checkpoint-by-metric
- evaluation hooks computing Pearson/Spearman correlation on a validation split
- graceful failure with exit code 2 when heavy libs (torch>=2.1, transformers, datasets) are missing

Notes:
- Expects `datasets_for_training/pairwise_prefs.csv` with columns: question, preferred_answer, other_answer, preference
- Writes checkpoints to `output_dir` + `reward_train_checkpoints/`.

This script is intended to be run on a machine with a configured HF environment.
"""
import argparse
import os
import sys
import yaml
import logging
from logging.handlers import RotatingFileHandler


def fail_missing_libs(msg=None):
    if msg:
        print(msg)
    print("Missing required heavy libraries for production HF reward training. Install `torch>=2.1`, `transformers`, and `datasets`.")
    sys.exit(2)


try:
    import math
    import random
    import csv
    import numpy as np
    import torch
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0]); tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        fail_missing_libs()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import pearsonr, spearmanr
except Exception:
    fail_missing_libs()


def setup_logger(logpath=None):
    logger = logging.getLogger("reward_trainer")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logpath:
        fh = RotatingFileHandler(logpath, maxBytes=10_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


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
        # pack prompt + answer as single text
        a = self.tokenizer(prompt + "\n" + pref, truncation=True, max_length=self.max_length, return_tensors="pt")
        b = self.tokenizer(prompt + "\n" + other, truncation=True, max_length=self.max_length, return_tensors="pt")
        label = 1 if r.get("preference") == "left" else 0
        return {"a": a, "b": b, "label": label}


def read_pairs(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("preferred_answer") or not r.get("other_answer"):
                continue
            out.append(r)
    return out


def collate_batch(batch):
    import torch
    a_input_ids = torch.cat([item["a"]["input_ids"] for item in batch], dim=0)
    a_attn = torch.cat([item["a"]["attention_mask"] for item in batch], dim=0)
    b_input_ids = torch.cat([item["b"]["input_ids"] for item in batch], dim=0)
    b_attn = torch.cat([item["b"]["attention_mask"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    return {"a_input_ids": a_input_ids, "a_attention_mask": a_attn, "b_input_ids": b_input_ids, "b_attention_mask": b_attn, "labels": labels}


def pairwise_loss(score_pref, score_other):
    import torch
    x = score_pref - score_other
    return -torch.log(torch.sigmoid(x) + 1e-12).mean()


def evaluate_model(model, tokenizer, rows, device, max_length=256):
    # Compute score difference for each pair and compute correlations with label
    model.eval()
    diffs = []
    labels = []
    with torch.no_grad():
        for r in rows:
            prompt = r.get("question", "")
            pref = r.get("preferred_answer", "")
            other = r.get("other_answer", "")
            a = tokenizer(prompt + "\n" + pref, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            b = tokenizer(prompt + "\n" + other, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out_a = model(**a).logits.squeeze(-1).cpu().numpy().item()
            out_b = model(**b).logits.squeeze(-1).cpu().numpy().item()
            diffs.append(out_a - out_b)
            labels.append(1 if r.get("preference") == "left" else 0)
    # correlation expects two continuous arrays; convert labels to -1/1 or use diffs vs labels
    try:
        pearson = pearsonr(diffs, labels)[0]
    except Exception:
        pearson = float("nan")
    try:
        spearman = spearmanr(diffs, labels)[0]
    except Exception:
        spearman = float("nan")
    return {"pearson": pearson, "spearman": spearman}


def save_checkpoint(model, tokenizer, outdir, epoch, metric=None):
    import os
    path = os.path.join(outdir, f"checkpoint_epoch{epoch}")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    if metric is not None:
        with open(os.path.join(path, "metric.txt"), "w", encoding="utf-8") as f:
            f.write(str(metric))
    return path


def train_from_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logpath = os.path.join(cfg.get("output_dir", "outputs"), "train_reward.log")
    logger = setup_logger(logpath)

    data_path = os.path.join(cfg.get("root", "."), "datasets_for_training", "pairwise_prefs.csv")
    if not os.path.exists(data_path):
        logger.error("Pairwise data not found: %s", data_path)
        return 3

    rows = read_pairs(data_path)
    random_seed = cfg.get("seed", 42)
    import random
    random.seed(random_seed)
    random.shuffle(rows)
    n = len(rows)
    val_frac = cfg.get("val_fraction", 0.1)
    nval = max(1, int(n * val_frac))
    val_rows = rows[:nval]
    train_rows = rows[nval:]

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("model_name", "bert-base-uncased"))
    model = AutoModelForSequenceClassification.from_pretrained(cfg.get("model_name", "bert-base-uncased"), num_labels=1)
    model.to(device)

    train_ds = PairwiseDataset(train_rows, tokenizer, max_length=cfg.get("max_length", 256))
    val_ds = val_rows
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, collate_fn=collate_batch)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-5))
    total_steps = len(dl) * cfg.get("epochs", 1)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=cfg.get("warmup_steps", 0), num_training_steps=total_steps)

    best_metric = -float("inf")
    ckpt_dir = os.path.join(cfg.get("output_dir", "outputs"), "reward_train_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.get("epochs", 1) + 1):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(dl, start=1):
            optim.zero_grad()
            a = {"input_ids": batch["a_input_ids"].to(device), "attention_mask": batch["a_attention_mask"].to(device)}
            b = {"input_ids": batch["b_input_ids"].to(device), "attention_mask": batch["b_attention_mask"].to(device)}
            out_a = model(**a).logits.squeeze(-1)
            out_b = model(**b).logits.squeeze(-1)
            loss = pairwise_loss(out_a, out_b)
            loss.backward()
            optim.step()
            scheduler.step()
            running_loss += loss.item()
            if i % cfg.get("log_every", 10) == 0:
                logger.info("Epoch %d step %d loss=%.6f", epoch, i, running_loss / i)

        # evaluation
        metrics = evaluate_model(model, tokenizer, val_ds, device, max_length=cfg.get("max_length", 256))
        logger.info("Epoch %d eval: pearson=%.4f spearman=%.4f", epoch, metrics.get("pearson"), metrics.get("spearman"))

        # checkpoint
        ckpt_path = save_checkpoint(model, tokenizer, ckpt_dir, epoch, metric=metrics.get("pearson"))
        if metrics.get("pearson") and metrics.get("pearson") > best_metric:
            best_metric = metrics.get("pearson")
            best_path = os.path.join(cfg.get("output_dir", "outputs"), "best_reward_model")
            # copy latest ckpt to best
            import shutil
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(ckpt_path, best_path)
            logger.info("Saved best model to %s (pearson=%.4f)", best_path, best_metric)

    # final save
    final_path = os.path.join(cfg.get("output_dir", "outputs"), "final_reward_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Training complete. Final model saved to %s", final_path)
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reward_train.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}")
        return 2
    try:
        return train_from_config(cfg_path)
    except SystemExit:
        raise
    except Exception as e:
        print("Training failed:", e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
