import json
import os
import sys
sys.path.insert(0, os.getcwd())
from src.metrics_tracker import MetricsTracker

def recompute(detailed_metrics_path: str):
    if not os.path.exists(detailed_metrics_path):
        print('Could not find', detailed_metrics_path)
        return

    with open(detailed_metrics_path, 'r', encoding='utf-8') as fh:
        records = json.load(fh)

    mt = MetricsTracker(output_dir='outputs')

    # We'll recompute codebleu and ruby for each record if generated_texts and prompts exist
    new_records = []
    for r in records:
        prompts = r.get('prompt')
        gen = r.get('generated_code') or r.get('generated_texts')

        # Normalize to lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(gen, str):
            gen = [gen]

        if prompts is None or gen is None:
            # keep existing
            new_records.append(r)
            continue

        # Use prompts as references (best-effort fallback)
        refs = prompts

        try:
            metrics = mt.calculate_metrics(prompts, gen, refs)
            r.update(metrics)
        except Exception as e:
            print('Failed to compute metrics for a record:', e)

        new_records.append(r)

    # Save back
    with open(detailed_metrics_path, 'w', encoding='utf-8') as fh:
        json.dump(new_records, fh, indent=2, ensure_ascii=False)

    print('Recomputed metrics and updated', detailed_metrics_path)

if __name__ == '__main__':
    recompute(os.path.join('outputs', 'detailed_metrics.json'))
