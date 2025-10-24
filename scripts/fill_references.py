#!/usr/bin/env python3
import os
import re
import pandas as pd
from collections import Counter

DATA_DIR = 'datasets_for_eval'
EVAL_DETAILED = 'evaluation_results/final_evaluation_detailed.csv'

def tokenize(s: str):
    if s is None or s != s:
        return []
    s = re.sub(r"[`\"'\\\[\]{}()<>.,:;!\\/?@#\$%\^&\*\-_=+\\|~]", ' ', str(s).lower())
    toks = [t for t in s.split() if t]
    return toks

def overlap_score(a, b):
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = sa & sb
    # normalize by smaller length to be permissive
    denom = min(len(sa), len(sb))
    return len(inter) / denom if denom > 0 else 0.0

def find_reference_for_prompt(prompt_tokens, dfs_files):
    # search each csv file in chunks
    ref_cols = ['Answer','ModelAnswer','FullModelAnswer','snippet','ans_gt','ans','reference']
    search_cols = ['Prompt','Question','prompt','question','text','clean_question','instruction','input']
    def is_code_like(s: str) -> bool:
        if s is None or s != s:
            return False
        s = str(s)
        if '\n' in s and len(s.splitlines()) > 1:
            return True
        code_tokens = ['def ', 'return ', 'import ', 'class ', 'print(', 'self.', ':']
        hits = sum(1 for t in code_tokens if t in s)
        return hits >= 2
    for f in dfs_files:
        path = os.path.join(DATA_DIR, f)
        try:
            for chunk in pd.read_csv(path, chunksize=500, encoding='utf-8'):
                # for each search col present
                for sc in search_cols:
                    if sc in chunk.columns:
                        texts = chunk[sc].astype(str).fillna('')
                        for idx, text in texts.items():
                            t_tokens = tokenize(text)
                            score = overlap_score(prompt_tokens, t_tokens)
                            if score >= 0.5:
                                # find a ref column with non-empty value
                                # prefer code-like references
                                for rc in ref_cols:
                                    if rc in chunk.columns:
                                        val = chunk.at[idx, rc]
                                        if pd.notna(val) and str(val).strip() and is_code_like(val):
                                            return str(val)
                                # if none code-like, keep first non-empty as last resort
                                for rc in ref_cols:
                                    if rc in chunk.columns:
                                        val = chunk.at[idx, rc]
                                        if pd.notna(val) and str(val).strip():
                                            return str(val)
        except Exception:
            continue
    return None


def main():
    if not os.path.exists(EVAL_DETAILED):
        print('Evaluation detailed CSV not found:', EVAL_DETAILED)
        return

    df = pd.read_csv(EVAL_DETAILED)
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    filled = 0
    for i, row in df.iterrows():
        if pd.notna(row.get('reference')) and str(row.get('reference')).strip():
            continue
        prompt = row.get('prompt')
        tokens = tokenize(prompt)
        if not tokens:
            continue
        ref = find_reference_for_prompt(tokens, files)
        if ref:
            df.at[i, 'reference'] = ref
            filled += 1
            print(f'Filled row {i} with reference (len {len(ref)} chars)')

    print('Filled', filled, 'references out of', len(df))
    df.to_csv(EVAL_DETAILED, index=False)
    print('Saved updated', EVAL_DETAILED)

if __name__ == '__main__':
    main()
