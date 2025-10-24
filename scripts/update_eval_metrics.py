#!/usr/bin/env python3
import os
import pandas as pd
from src.metrics_tracker import MetricsTracker

EVAL_DETAILED = 'evaluation_results/final_evaluation_detailed.csv'


def is_code_like(text: str) -> bool:
    if text is None or text != text:
        return False
    s = str(text)
    # heuristics: presence of typical code tokens or multiple lines
    tokens = ['def ', 'return ', 'import ', 'from ', 'class ', '\n', '():', '):', 'print(', 'self.', ':']
    hits = sum(1 for t in tokens if t in s)
    # multi-line or at least two token hits
    return ('\n' in s and len(s.splitlines()) > 1) or hits >= 2


def main():
    if not os.path.exists(EVAL_DETAILED):
        print('File not found:', EVAL_DETAILED)
        return

    df = pd.read_csv(EVAL_DETAILED)
    mt = MetricsTracker(output_dir='outputs')

    updated = 0
    for i, row in df.iterrows():
        prompt = row.get('prompt')
        gen = row.get('generated_code')
        ref = row.get('reference')
        if not pd.isna(ref) and is_code_like(ref):
            # compute metrics
            try:
                metrics = mt.calculate_metrics([prompt], [gen], [ref])
                for k, v in metrics.items():
                    df.at[i, k] = v
                updated += 1
            except Exception as e:
                print('Failed metrics for row', i, e)
        else:
            # attempt to mark codebleu/ruby as NaN explicitly to show missing reference or non-code ref
            df.at[i, 'codebleu'] = 0.0
            df.at[i, 'ruby'] = 0.0

    print('Updated metrics for', updated, 'rows out of', len(df))
    df.to_csv(EVAL_DETAILED, index=False)
    print('Saved updated', EVAL_DETAILED)


if __name__ == '__main__':
    main()
