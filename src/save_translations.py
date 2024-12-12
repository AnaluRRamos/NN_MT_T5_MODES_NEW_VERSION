import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import T5TokenizerFast
from model import MT5FineTuner
from preprocess import MT5Dataset, custom_collate_fn
from torch.utils.data import DataLoader
from config import Config

# Tokenizer setup
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

# Dataset setup
data_dir = Config.DATA_DIR
max_len = Config.TARGET_MAX_LENGTH

train_dataset = MT5Dataset(
    data_dir=os.path.join(data_dir, 'train'),
    source_ext='_en.txt',
    target_ext='_pt.txt',
    tokenizer=tokenizer,
    max_len=max_len
)

val_dataset = MT5Dataset(
    data_dir=os.path.join(data_dir, 'val'),
    source_ext='_en.txt',
    target_ext='_pt.txt',
    tokenizer=tokenizer,
    max_len=max_len
)

test_dataset = MT5Dataset(
    data_dir=os.path.join(data_dir, 'test'),
    source_ext='_en.txt',
    target_ext='_pt.txt',
    tokenizer=tokenizer,
    max_len=max_len
)

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=Config.BATCH_SIZE, 
    shuffle=False, 
    collate_fn=custom_collate_fn
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=Config.BATCH_SIZE, 
    shuffle=False, 
    collate_fn=custom_collate_fn
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=Config.BATCH_SIZE, 
    shuffle=False, 
    collate_fn=custom_collate_fn
)

# Load checkpoint with dataloader arguments
checkpoint_path = "/home/strwbrryanalu/NN_MT_T5_WITH_MODES_NEWVERSION/output/checkpoints/t5_finetuner-epoch=09-val_loss=11.22.ckpt"

model = MT5FineTuner.load_from_checkpoint(
    checkpoint_path,
    tokenizer=tokenizer,
    learning_rate=Config.LEARNING_RATE,
    target_max_length=Config.TARGET_MAX_LENGTH,
    mode=Config.MODE,
    num_ne_tags=26,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader
)

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_translations(model, dataset, dataset_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    translations = []
    references = []

    for batch in dataloader:
        source_token_ids, source_mask, source_ne_tags, target_token_ids, target_mask = batch

        source_token_ids = source_token_ids.to(device)
        source_mask = source_mask.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=source_token_ids,
                attention_mask=source_mask,
                max_length=Config.TARGET_MAX_LENGTH,
                num_beams=4,
                early_stopping=True
            )

        # Decode first item in batch (you might want to handle multiple items)
        predicted_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        reference_text = tokenizer.decode(
            target_token_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        translations.append(predicted_text)
        references.append(reference_text)

    save_translations(translations, references, dataset_name)

def save_translations(translations, references, dataset_name):
    translations_dir = 'translations'
    os.makedirs(translations_dir, exist_ok=True)

    translations_file = os.path.join(translations_dir, f"{dataset_name}_translations.txt")
    references_file = os.path.join(translations_dir, f"{dataset_name}_references.txt")

    with open(translations_file, 'w', encoding='utf-8') as f:
        for line in translations:
            f.write(line + '\n')

    with open(references_file, 'w', encoding='utf-8') as f:
        for line in references:
            f.write(line + '\n')

    print(f"Saved {len(translations)} translations for {dataset_name} dataset in '{translations_dir}' directory.")


generate_translations(model, train_dataset, 'train')
generate_translations(model, val_dataset, 'val')
generate_translations(model, test_dataset, 'test')