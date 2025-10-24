import json
import os
from collections import defaultdict

def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def aggregate_by_epoch(records):
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for r in records:
        e = int(r.get('epoch', 0))
        counts[e] += 1
        for key in ['reward', 'bertscore', 'bleu', 'codebleu', 'rouge', 'syntax_score', 'structure_score']:
            if key in r and r[key] is not None:
                try:
                    sums[e][key] += float(r[key])
                except Exception:
                    pass

    epochs = sorted(counts.keys())
    out = []
    for e in epochs:
        row = {'epoch': e}
        for key in ['reward', 'bertscore', 'bleu', 'codebleu', 'rouge', 'syntax_score', 'structure_score']:
            row[key] = (sums[e].get(key, 0.0) / counts[e]) if counts[e] > 0 else 0.0
        out.append(row)
    return out

def save_csv(agg, out_path):
    import csv
    keys = ['epoch','reward','syntax_score','structure_score','bertscore','bleu','codebleu','rouge']
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in agg:
            writer.writerow({k: r.get(k, '') for k in keys})

def plot(agg, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('matplotlib not available, will not plot PNG:', e)
        return False

    epochs = [r['epoch'] for r in agg]
    def series(k):
        return [r.get(k, 0.0) for r in agg]

    plt.figure(figsize=(10,6))
    plt.plot(epochs, series('reward'), label='reward')
    plt.plot(epochs, series('bertscore'), label='bertscore')
    plt.plot(epochs, series('bleu'), label='bleu')
    plt.plot(epochs, series('codebleu'), label='codebleu')
    plt.plot(epochs, series('rouge'), label='rouge')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    return True

if __name__ == '__main__':
    metrics_path = os.path.join('outputs', 'detailed_metrics.json')
    if not os.path.exists(metrics_path):
        print('Could not find', metrics_path)
        raise SystemExit(1)

    records = load_metrics(metrics_path)
    agg = aggregate_by_epoch(records)

    os.makedirs('outputs', exist_ok=True)
    csv_out = os.path.join('outputs', 'metrics_by_epoch.csv')
    save_csv(agg, csv_out)
    print('Saved CSV to', csv_out)

    png_out = os.path.join('outputs', 'metrics_by_epoch.png')
    ok = plot(agg, png_out)
    if ok:
        print('Saved plot to', png_out)
    else:
        print('Plot not created')
