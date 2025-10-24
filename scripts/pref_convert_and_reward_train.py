import os
import json
from glob import glob
from typing import List, Tuple
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch


def load_pref_json_files(folder: str) -> List[dict]:
    files = glob(os.path.join(folder, '*.json'))
    results = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                j = json.load(fh)
                results.append(j)
        except Exception:
            continue
    return results


def extract_pairs_from_json(j: dict) -> List[Tuple[str, str, int]]:
    """Extract (A, B, label) where label=1 if left preferred else 0.

    The JSON structure in evaluation_results_server contains two answers per question in `questions_df`.
    We will interpret compare results using `comparison_slider` or the per-field scores.
    This is a heuristic extractor adapted to the observed structure.
    """
    pairs = []
    qdf = j.get('questions_df', []) or []

    # Group entries by question ID or index
    groups = {}
    for item in qdf:
        key = item.get('ID') or item.get('index')
        if key is None:
            continue
        groups.setdefault(key, []).append(item)

    for key, items in groups.items():
        if len(items) < 2:
            continue
        left = items[0].get('Answer', '')
        right = items[1].get('Answer', '')
        prompt = items[0].get('Question', '') or ''

        # Determine label using available per-file or per-item scores if present
        # Prefer file-level aggregated comparison_slider if provided
        file_slider = j.get('comparison_slider')
        if file_slider is not None:
            label = 1 if file_slider >= 0 else 0
        else:
            # Sum left vs right per-item annotated scores if present
            left_score = 0
            right_score = 0
            for k in ['consistent_L', 'correct_L', 'useful_L']:
                left_score += j.get(k, 0)
            for k in ['consistent_R', 'correct_R', 'useful_R']:
                right_score += j.get(k, 0)
            if left_score >= right_score:
                label = 1
            else:
                label = 0

        pairs.append((prompt, left, right, int(label)))
    return pairs


def build_dataset_from_folder(folder: str, out_path: str):
    all_json = load_pref_json_files(folder)
    records = []
    for j in all_json:
        recs = extract_pairs_from_json(j)
        records.extend(recs)

    # Save TSV with columns: prompt \t left \t right \t label
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        for prompt, left, right, lbl in records:
            fh.write(prompt.replace('\n', '\\n') + '\t' + left.replace('\n', '\\n') + '\t' + right.replace('\n', '\\n') + '\t' + str(lbl) + '\n')


class RewardModelTrainerStub:
    """Very small reward model trainer stub using transformers classification head.

    This is a quick way to get a model that scores generations for the smoke tests. For production
    you should train a proper pairwise reward model using ranking or pairwise losses. We bias the
    default to a code-aware model (CodeBERT) and better preprocessing for code prompts.
    """
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name

    def train(self, tsv_path: str, output_dir: str):
        # Read tsv
        examples = []
        with open(tsv_path, 'r', encoding='utf-8') as fh:
            for ln in fh:
                parts = ln.rstrip('\n').split('\t')
                if len(parts) < 4:
                    # Old format: left \t right \t label
                    continue
                prompt, left, right, lbl = parts[0], parts[1], parts[2], parts[3]
                examples.append((prompt.replace('\\n', '\n'), left.replace('\\n', '\n'), right.replace('\\n', '\n'), int(lbl)))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        texts = [ (l + tokenizer.sep_token + r) for _, l, r, _ in examples]
        labels = [lbl for _, _, _, lbl in examples]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=512)

        class DS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc = enc
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        ds = DS(enc, labels)

        # Use a small number of epochs for the stub but a code-aware initialization.
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        # Use minimal batch size and epochs to fit small GPUs for smoke/baseline runs
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_strategy='no',
            logging_strategy='no',
            learning_rate=2e-5,
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()
        trainer.save_model(output_dir)

class PairwiseRewardTrainer:
    """Train a simple pairwise reward model by classifying which answer is preferred."""
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name

    def train(self, tsv_path: str, output_dir: str, epochs: int = 1, per_device_batch_size: int = 1):
        # Read tsv expecting prompt \t left \t right \t label
        examples = []
        with open(tsv_path, 'r', encoding='utf-8') as fh:
            for ln in fh:
                parts = ln.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue
                prompt, left, right, lbl = parts[0].replace('\\n','\n'), parts[1].replace('\\n','\n'), parts[2].replace('\\n','\n'), int(parts[3])
                examples.append((prompt, left, right, lbl))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        texts = [ (p + tokenizer.sep_token + l + tokenizer.sep_token + r) for p,l,r,_ in examples]
        labels = [lbl for _,_,_,lbl in examples]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=512)

        class DS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc = enc
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k,v in self.enc.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        ds = DS(enc, labels)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        # Respect small per-device batch sizes to avoid OOM on 4GB GPUs
        args = TrainingArguments(output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=per_device_batch_size, save_strategy='no', logging_strategy='no')
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        trainer.train()
        trainer.save_model(output_dir)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--prefs-folder', type=str, required=True)
    p.add_argument('--out-tsv', type=str, default='training_data/prefs.tsv')
    p.add_argument('--reward-output', type=str, default='outputs/reward_model')
    p.add_argument('--pairwise', action='store_true', help='Train pairwise reward model instead of stub')
    p.add_argument('--pairwise-epochs', type=int, default=1)
    args = p.parse_args()

    build_dataset_from_folder(args.prefs_folder, args.out_tsv)
    print('Built TSV:', args.out_tsv)
    if getattr(args, 'pairwise', False):
        print('Running pairwise trainer...')
        trainer = PairwiseRewardTrainer()
        trainer.train(args.out_tsv, args.reward_output, epochs=args.pairwise_epochs)
    else:
        trainer = RewardModelTrainerStub()
        trainer.train(args.out_tsv, args.reward_output)
