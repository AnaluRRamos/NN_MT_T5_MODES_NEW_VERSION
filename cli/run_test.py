import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from src.evaluate import evaluate_model
from transformers import MT5TokenizerFast
from src.model import MT5FineTuner
from src.utils import load_data
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Test the MT5 model on labeled test data")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-small')
    model = MT5FineTuner.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer, mode=Config.MODE)

    _, _, test_dataloader = load_data(Config.DATA_DIR, tokenizer, Config.BATCH_SIZE)

    print("Running test evaluation...")
    bleu_score = evaluate_model(model, test_dataloader)
    print(f"BLEU Score on test set: {bleu_score}")

if __name__ == "__main__":
    main()
