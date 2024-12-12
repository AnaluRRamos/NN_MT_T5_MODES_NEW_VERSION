import sacrebleu
import torch
from transformers import MT5TokenizerFast

def evaluate_model(model, dataloader):
    tokenizer = model.tokenizer
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictions, references = [], []

    for batch in dataloader:
        source_ids, source_mask, target_ids, _, _ = batch
        source_ids = source_ids.to(model.device)
        source_mask = source_mask.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=model.target_max_length)
        
        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
        #pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        #ref_texts = tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        predictions.extend(pred_texts)
        references.extend(ref_texts)
    
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    #rouge_score = calculate_rouge(predictions, references)  # Hypothetical function for ROUGE
    #print(f"BLEU Score: {bleu_score}, ROUGE Score: {rouge_score}")
    #return {'bleu': bleu_score, 'rouge': rouge_score}
    # Inside evaluate_model function
    for i, (pred, ref) in enumerate(zip(predictions[:5], references[:5])):  # Show first 5 pairs
        print(f"Sample {i + 1} Prediction: {pred}")
        print(f"Sample {i + 1} Reference: {ref}")

    print(f"BLEU Score: {bleu_score}")

    return bleu_score

#for i, batch in enumerate(dataloader):
    # Usual process here
    #if i % 10 == 0:  # Log every 10 batches or as desired
        #print(f"Evaluating batch {i + 1} out of {len(dataloader)}")
