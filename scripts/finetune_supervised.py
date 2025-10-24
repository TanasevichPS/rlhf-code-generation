"""Lightweight supervised finetune placeholder.

Reads `datasets_for_training/sft_dataset.csv` and creates a tiny "model"
artifact that contains a simple vocabulary and example mappings. This file
is meant as a fast smoke-test for the SFT pipeline and to be replaced by a
real HuggingFace training run once the environment is ready.

Outputs:
    outputs/sft_model_placeholder.json
"""
import csv
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(ROOT, "datasets_for_training", "sft_dataset.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "sft_model_placeholder.json")


def load_sft(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_vocab(texts, max_vocab=2000):
    freq = {}
    for t in texts:
        for tok in (t or "").split():
            freq[tok] = freq.get(tok, 0) + 1
    items = sorted(freq.items(), key=lambda x: -x[1])[:max_vocab]
    vocab = {tok: idx for idx, (tok, _) in enumerate(items, start=1)}
    vocab["<unk>"] = 0
    return vocab


def main():
    if not os.path.exists(IN_PATH):
        logging.error("Required input not found: %s", IN_PATH)
        logging.error("Run scripts/prepare_pairs.py first to generate datasets.")
        return 2

    rows = load_sft(IN_PATH)
    if not rows:
        logging.error("No rows found in %s", IN_PATH)
        return 2

    questions = [r.get("question", "") for r in rows]
    answers = [r.get("best_answer", "") for r in rows]

    vocab = build_vocab(questions + answers, max_vocab=2000)

    artifact = {"meta": {"type": "placeholder_sft", "rows": len(rows)}, "vocab_size": len(vocab), "vocab_sample": dict(list(vocab.items())[:20])}
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    logging.info("Saved SFT placeholder model to %s (vocab size=%d)", OUT_PATH, len(vocab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
