from modern_rlhf import ModernRLHFPipeline, get_research_config
import traceback

def main():
    try:
        cfg = get_research_config()
        # Apply the same config overrides as run_modern_rlhf.py to reproduce behavior
        cfg.data.train_data_path = "./datasets_for_training"
        cfg.data.eval_data_path = "./datasets_for_eval"
        cfg.data.human_feedback_path = "./evaluation_results_server"
        cfg.data.output_path = "./modern_outputs"
        cfg.data.min_prompt_length = 0
        cfg.data.min_response_length = 0
        cfg.data.conala_local_path = r"C:\Users\Полина\Desktop\Работа\huawei\rlhf\conala-corpus"
        cfg.experiment_name = "modern_rlhf_experiment"
        cfg.training.ppo_epochs = 3
        cfg.training.total_steps = 500
        cfg.evaluation.eval_samples = 50
        cfg.training.learning_rate = 1e-5
        p = ModernRLHFPipeline(cfg)
        train_data, eval_data, hf = p.load_data()
        print('Train samples count:', len(train_data))
        if train_data:
            print('First sample type:', type(train_data[0]))
        try:
            batches = p._prepare_reward_training_batches(train_data)
            print('Prepared batches:', len(batches))
        except Exception:
            print('Error while preparing reward batches:')
            traceback.print_exc()

        try:
            print('Running reward model preparation to capture errors...')
            p.prepare_reward_model(train_data, hf)
            print('Reward model prepared successfully')
        except Exception:
            print('Error during reward model preparation:')
            traceback.print_exc()

        try:
            print('Preparing RLHF trainer...')
            p.prepare_rlhf_trainer()
            print('RLHF trainer prepared')
        except Exception:
            print('Error preparing RLHF trainer:')
            traceback.print_exc()

        try:
            print('Starting RLHF training...')
            tm = p.train_rlhf(train_data, [])
            print('RLHF training completed, metrics:', tm)
        except Exception:
            print('Error during RLHF training:')
            traceback.print_exc()

        try:
            print('Evaluating model...')
            em = p.evaluate_model([])
            print('Evaluation metrics:', em)
        except Exception:
            print('Error during evaluation:')
            traceback.print_exc()
        except Exception:
            print('Error while preparing reward batches:')
            traceback.print_exc()
    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    main()


