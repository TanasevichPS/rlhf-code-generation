"""
Supervised fine-tuning for causal models using HuggingFace Transformers.

This script consumes `datasets_for_training/sft_dataset.csv` (question,best_answer)
and runs a standard causal LM finetuning using `Trainer` and `DataCollatorForLanguageModeling`.

If required libraries are missing, exits with code 2 (so CI/tests can assert).
"""
import argparse
import os
import sys
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fail_missing_libs():
    print("Missing required libraries for HF SFT. Install `torch` and `transformers`.")
    sys.exit(2)


try:
    import torch
    tver = getattr(torch, "__version__", "0.0.0")
    try:
        ver_parts = tver.split("+")[0].split(".")
        tmajor = int(ver_parts[0])
        tminor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
    except Exception:
        tmajor, tminor = 0, 0
    if (tmajor, tminor) < (2, 1):
        fail_missing_libs()

    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from transformers import DataCollatorForLanguageModeling
    from torch.utils.data import Dataset
except Exception:
    fail_missing_libs()


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.examples = []
        for r in rows:
            q = r.get("question", "")
            a = r.get("best_answer", "")
            text = (q + "\n" + a).strip()
            self.examples.append(tokenizer(text, truncation=True, max_length=max_length))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


def read_sft(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("best_answer"):
                continue
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--output_dir", default="outputs/sft_model_hf")
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    data_path = os.path.join(args.root, "datasets_for_training", "sft_dataset.csv")
    if not os.path.exists(data_path):
        logging.error("SFT dataset not found: %s", data_path)
        return 3

    rows = read_sft(data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    ds = SFTDataset(rows, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    logging.info("Saved SFT model to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
