import json
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def process_human_evaluations(json_dir: str) -> pd.DataFrame:
    """Process human evaluations for reward model training."""
    evaluations = []
    json_files = Path(json_dir).glob("*.json")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for q in data.get('questions_df', []):
                # Normalize scores from -2..2 to 0..1
                def normalize_score(score):
                    return (score + 2) / 4.0
                
                evaluation = {
                    'question': q['Question'],
                    'answer': q['Answer'],
                    'model_tag': q['MODEL_TAG'],
                    'consistent_score': normalize_score(q.get('consistent_L', 0) + q.get('consistent_R', 0)),
                    'correct_score': normalize_score(q.get('correct_L', 0) + q.get('correct_R', 0)),
                    'useful_score': normalize_score(q.get('useful_L', 0) + q.get('useful_R', 0)),
                    'total_score': normalize_score(
                        q.get('consistent_L', 0) + q.get('consistent_R', 0) +
                        q.get('correct_L', 0) + q.get('correct_R', 0) +
                        q.get('useful_L', 0) + q.get('useful_R', 0)
                    )
                }
                evaluations.append(evaluation)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(evaluations)
    logger.info(f"Processed {len(df)} human evaluations")
    return df