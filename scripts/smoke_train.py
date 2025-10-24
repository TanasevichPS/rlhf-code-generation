import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch


def create_small_dataset():
    # A tiny synthetic dataset of prompt -> reference code (python)
    return [
        ("Write a function that returns the sum of a list of integers.", "def sum_list(lst):\n    return sum(lst)\n"),
        ("Check if a string is a palindrome.", "def is_pal(s):\n    s2 = s.replace(' ', '').lower()\n    return s2 == s2[::-1]\n"),
    ]


def run_sft_one_epoch(model_name: str, output_dir: str):
    data = create_small_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    texts = ["Prompt: " + p + "\nCode: " + r for p, r in data]
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return self.enc['input_ids'].size(0)

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}

    ds = DS(enc)

    args = TrainingArguments(output_dir=output_dir, num_train_epochs=1, per_device_train_batch_size=1, save_strategy='no', logging_steps=1)
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--out', default='outputs/sft_model')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_sft_one_epoch(args.model, args.out)
    print('SFT smoke training done. Model saved to', args.out)


if __name__ == '__main__':
    main()
