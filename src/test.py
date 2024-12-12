import os
import argparse
from src.utils import load_data
from src.model import MT5FineTuner
from src.evaluate import evaluate_model

def load_test_data(test_path):
    with open(test_path, 'r') as f:
        data = f.readlines()
    return [line.strip() for line in data]

def test_model(model, test_data, mode):
    predictions = []
    for text in test_data:
        prediction = model.translate(text, mode=mode)
        predictions.append(prediction)
    return predictions

def main(test_path, checkpoint_path, mode):
    test_data = load_test_data(test_path)
    model = MT5FineTuner.load_from_checkpoint(checkpoint_path, mode=mode)
    predictions = test_model(model, test_data, mode)
    evaluation_results = evaluate_model(predictions, test_data, mode)
    print(f"Results for {mode}: {evaluation_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on the test dataset.")
    parser.add_argument('--test_path', type=str, default='data/test/medline_en2pt_en.txt', help="Path to the test file.")
    parser.add_argument('--checkpoint_path', type=str, default='output/checkpoints/best_model.ckpt', help="Path to the model checkpoint.")
    parser.add_argument('--mode', type=str, default='mode_0', help="Mode for model evaluation.")
    args = parser.parse_args()
    main(args.test_path, args.checkpoint_path, args.mode)
