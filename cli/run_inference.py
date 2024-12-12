import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from transformers import MT5TokenizerFast
from src.model import MT5FineTuner

def main():
    parser = argparse.ArgumentParser(description="Run inference using the MT5 model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_text", type=str, required=True, help="Input text for translation")
    args = parser.parse_args()

    tokenizer = MT5TokenizerFast.from_pretrained('google/mt5-small')
    model = MT5FineTuner.load_from_checkpoint(args.checkpoint_path, tokenizer=tokenizer)

    print("Running inference...")
    output_text = model.translate(args.input_text)
    print(f"Translation: {output_text}")

if __name__ == "__main__":
    main()
