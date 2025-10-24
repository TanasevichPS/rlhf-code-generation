"""
Prepare pairwise preference and SFT datasets from human comparison JSONs.

This script scans the `evaluation_results_server/` directory for JSON files
containing human comparisons. For each record it expects a top-level set of
ratings for left/right (e.g., `consistent_L`, `correct_L`, `useful_L`) and a
`questions_df` list with two entries (left then right) holding `Question` and
`Answer` and metadata. It computes which side is preferred by summing the
left/right scores and emits two CSVs:

- datasets_for_training/pairwise_prefs.csv:
    question,preferred_answer,other_answer,preferred_model_tag,other_model_tag,preference,source_json,datetime

- datasets_for_training/sft_dataset.csv:
    question,best_answer,model_tag,source_json,datetime

Assumptions made when converting:
- `questions_df` contains exactly two entries: index 0 -> LEFT, index 1 -> RIGHT.
- Preference is decided by sum(consistent, correct, useful) for left vs right.
- Ties are skipped in pairwise output but not in SFT.

If some files don't match the expected format the script will log warnings and
skip those entries.
"""
import os
import glob
import json
import csv
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "evaluation_results_server")
OUT_DIR = os.path.join(ROOT, "datasets_for_training")
os.makedirs(OUT_DIR, exist_ok=True)

PAIRWISE_OUT = os.path.join(OUT_DIR, "pairwise_prefs.csv")
SFT_OUT = os.path.join(OUT_DIR, "sft_dataset.csv")


def parse_record(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dt = data.get("datetime")
    src = os.path.basename(path)

    # get top-level L/R scores
    try:
        score_L = sum(int(data.get(k, 0)) for k in ("consistent_L", "correct_L", "useful_L"))
        score_R = sum(int(data.get(k, 0)) for k in ("consistent_R", "correct_R", "useful_R"))
    except Exception:
        logging.warning("Invalid score fields in %s", src)
        return []

    qlist = data.get("questions_df") or []
    if not isinstance(qlist, list) or len(qlist) < 2:
        logging.warning("Skipping %s: questions_df missing or <2 entries", src)
        return []

    # assume first is LEFT, second is RIGHT
    left = qlist[0]
    right = qlist[1]

    question = left.get("Question") or right.get("Question") or ""
    left_ans = left.get("Answer", "")
    right_ans = right.get("Answer", "")
    left_tag = left.get("MODEL_TAG", left.get("CSV_PATH", "left"))
    right_tag = right.get("MODEL_TAG", right.get("CSV_PATH", "right"))

    preferred = None
    if score_L > score_R:
        preferred = "left"
    elif score_R > score_L:
        preferred = "right"
    else:
        preferred = "tie"

    recs = []
    # pairwise entry (skip ties)
    if preferred != "tie":
        if preferred == "left":
            recs.append({
                "question": question,
                "preferred_answer": left_ans,
                "other_answer": right_ans,
                "preferred_model_tag": left_tag,
                "other_model_tag": right_tag,
                "preference": "left",
                "source_json": src,
                "datetime": dt,
            })
        else:
            recs.append({
                "question": question,
                "preferred_answer": right_ans,
                "other_answer": left_ans,
                "preferred_model_tag": right_tag,
                "other_model_tag": left_tag,
                "preference": "right",
                "source_json": src,
                "datetime": dt,
            })

    # SFT entry: include best answer even if tie (choose left in ties)
    best = left_ans if (preferred != "right") else right_ans
    best_tag = left_tag if (preferred != "right") else right_tag
    recs.append({
        "question": question,
        "best_answer": best,
        "model_tag": best_tag,
        "source_json": src,
        "datetime": dt,
    })

    return recs


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not files:
        logging.error("No JSON files found in %s", INPUT_DIR)
        return

    pairwise_fields = [
        "question",
        "preferred_answer",
        "other_answer",
        "preferred_model_tag",
        "other_model_tag",
        "preference",
        "source_json",
        "datetime",
    ]

    sft_fields = ["question", "best_answer", "model_tag", "source_json", "datetime"]

    pair_count = 0
    sft_count = 0

    with open(PAIRWISE_OUT, "w", encoding="utf-8", newline="") as pf, open(SFT_OUT, "w", encoding="utf-8", newline="") as sf:
        pw = csv.DictWriter(pf, fieldnames=pairwise_fields)
        sw = csv.DictWriter(sf, fieldnames=sft_fields)
        pw.writeheader()
        sw.writeheader()

        for p in files:
            recs = parse_record(p)
            for r in recs:
                # pairwise rows have 'preferred_answer' key
                if "preferred_answer" in r:
                    pw.writerow(r)
                    pair_count += 1
                else:
                    sw.writerow(r)
                    sft_count += 1

    logging.info("Wrote %d pairwise rows to %s", pair_count, PAIRWISE_OUT)
    logging.info("Wrote %d SFT rows to %s", sft_count, SFT_OUT)


if __name__ == "__main__":
    main()
