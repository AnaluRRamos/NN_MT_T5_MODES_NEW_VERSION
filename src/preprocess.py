import os
import glob
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import MT5TokenizerFast, AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

class MT5Dataset(Dataset):
    def __init__(self, data_dir, source_ext='_en.txt', target_ext='_pt.txt', tokenizer=None, max_len=512):
        self.source_files = sorted(glob.glob(os.path.join(data_dir, f"*{source_ext}")))
        self.target_files = sorted(glob.glob(os.path.join(data_dir, f"*{target_ext}")))

        assert len(self.source_files) == len(self.target_files), "Mismatch between source and target files."
        for src, tgt in zip(self.source_files[:5],self.target_files[:5]):
            print(f"Source: {os.path.basename(src)}, Target: {os.path.basename(tgt)}")
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Number of matched English-Portuguese pairs: {len(self.source_files)}")
        
        try:
            self.nlp_spacy = spacy.load("en_ner_bionlp13cg_md")
        except Exception as e:
            raise ValueError("Ensure that the SpaCy model 'en_ner_bionlp13cg_md' is installed.") from e
        try:
            self.tokenizer_hf = AutoTokenizer.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.model_hf = AutoModelForTokenClassification.from_pretrained("Kushtrim/bert-base-cased-biomedical-ner")
            self.ner_pipeline = pipeline("ner", model=self.model_hf, tokenizer=self.tokenizer_hf, aggregation_strategy="simple")
        except Exception as e:
            raise ValueError("Ensure that the model 'Kushtrim/bert-base-cased-biomedical-ner' is available.") from e
        self.tag_to_idx = {
            'O': 0,
            'AMINO_ACID': 1,
            'ANATOMICAL_SYSTEM': 2,
            'CANCER': 3,
            'CELL': 4,
            'CELLULAR_COMPONENT': 5,
            'DEVELOPING_ANATOMICAL_STRUCTURE': 6,
            'GENE_OR_GENE_PRODUCT': 7,
            'IMMATERIAL_ANATOMICAL_ENTITY': 8,
            'MULTI_TISSUE_STRUCTURE': 9,
            'ORGAN': 10,
            'ORGANISM_SPACY': 11,
            'ORGANISM_SUBDIVISION': 12,
            'ORGANISM_SUBSTANCE': 13,
            'PATHOLOGICAL_FORMATION': 14,
            'SIMPLE_CHEMICAL': 15,
            'TISSUE_SPACY': 16,
            'SMALL_MOLECULE': 17,
            'GENEPROD': 18,
            'SUBCELLULAR': 19,
            'CELL_LINE': 20,
            'CELL_TYPE': 21,
            'TISSUE_HF': 22,
            'ORGANISM_HF': 23,
            'DISEASE': 24,
            'EXP_ASSAY': 25,
        }

    def __len__(self):
        return len(self.source_files)

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def combined_ne_tag(self, text):
        doc_spacy = self.nlp_spacy(text)
        entities_spacy = [(ent.start_char, ent.end_char, ent.label_) for ent in doc_spacy.ents]
        entities_spacy = [
            (start, end, label if label not in ['TISSUE', 'ORGANISM'] else f"{label}_SPACY")
            for start, end, label in entities_spacy
        ]
        ner_results = self.ner_pipeline(text)
        entities_hf = []
        for entity in ner_results:
            ent_start = entity['start']
            ent_end = entity['end']
            ent_label = entity['entity_group']
            if ent_label in ['TISSUE', 'ORGANISM']:
                ent_label += '_HF'
            entities_hf.append((ent_start, ent_end, ent_label))
        combined_entities = entities_spacy + entities_hf
        combined_entities = self.resolve_overlaps(combined_entities, entities_hf)
        return combined_entities

    def resolve_overlaps(self, entities, entities_hf):
        entities = sorted(entities, key=lambda x: (x[0], x[1]))
        resolved_entities = []
        for ent in entities:
            if not resolved_entities:
                resolved_entities.append(ent)
            else:
                last_ent = resolved_entities[-1]
                if ent[0] < last_ent[1]:
                    if ent in entities_hf:
                        resolved_entities[-1] = ent
                else:
                    resolved_entities.append(ent)
        return resolved_entities

    def preprocess(self, text):
        entities = self.combined_ne_tag(text)
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_attention_mask=True
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        offsets = tokenized_text['offset_mapping'].squeeze(0)
        aligned_ne_tags = self.align_ne_tags_with_tokens(text, entities, offsets, input_ids)
        assert input_ids.shape[0] == attention_mask.shape[0] == aligned_ne_tags.shape[0], \
            f"Shape mismatch: input_ids {input_ids.shape}, attention_mask {attention_mask.shape}, ne_tags {aligned_ne_tags.shape}"
        return input_ids, attention_mask, aligned_ne_tags

    def align_ne_tags_with_tokens(self, text, entities, offsets, input_ids):
        input_ids_list = input_ids.tolist()
        offsets_list = offsets.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids_list)
        assert len(tokens) == len(offsets_list), f"Tokens and offsets must have the same length, got {len(tokens)} and {len(offsets_list)}"
        assert len(tokens) == len(input_ids_list), f"Tokens and input_ids must have the same length, got {len(tokens)} and {len(input_ids_list)}"
        aligned_ne_tags = []
        for idx, (token, (start, end)) in enumerate(zip(tokens, offsets_list)):
            if start == end:
                aligned_ne_tags.append('O')
            else:
                tag = 'O'
                for ent_start, ent_end, ent_label in entities:
                    if (start >= ent_start and end <= ent_end) or (start < ent_end and end > ent_start):
                        tag = ent_label
                        break
                aligned_ne_tags.append(tag)
        aligned_ne_tag_ids = torch.tensor(
            [self.tag_to_idx.get(tag, 0) for tag in aligned_ne_tags],
            dtype=torch.long
        )
        input_length = input_ids.shape[0]
        if aligned_ne_tag_ids.size(0) < input_length:
            pad_length = input_length - aligned_ne_tag_ids.size(0)
            aligned_ne_tag_ids = torch.cat(
                [aligned_ne_tag_ids, torch.zeros(pad_length, dtype=torch.long)]
            )
        else:
            aligned_ne_tag_ids = aligned_ne_tag_ids[:input_length]
        assert aligned_ne_tag_ids.shape[0] == input_length, \
            f"NE tag ids length {aligned_ne_tag_ids.shape[0]} does not match input_ids length {input_length}"
        

        print("Debugging Token Alignment:")
        for idx, (token, (start, end)) in enumerate(zip(tokens, offsets_list)):
            tag = "O"
            for ent_start, ent_end, ent_label in entities:
                if (start >= ent_start and end <= ent_end) or (start < ent_end and end > ent_start):
                    tag = ent_label
                break
            print(f"Token: {token}, Offset: ({start}, {end}), Tag: {tag}")
        
        
        return aligned_ne_tag_ids

    def preprocess_target(self, text):
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True
        )
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        return input_ids, attention_mask

    def __getitem__(self, idx):
        source_text = self.load_file(self.source_files[idx])
        target_text = self.load_file(self.target_files[idx])

        source_text = f"translate English to Portuguese: {source_text}"
        
        source_input_ids, source_attention_mask, source_ne_tags = self.preprocess(source_text)
        target_input_ids, target_attention_mask = self.preprocess_target(target_text)
        return source_input_ids, source_attention_mask, source_ne_tags, target_input_ids, target_attention_mask

    def get_max_lengths(self):
        max_source_length = 0
        max_target_length = 0
        for source_file, target_file in zip(self.source_files, self.target_files):
            source_text = self.load_file(source_file)
            target_text = self.load_file(target_file)
            source_tokens = self.tokenizer.encode(source_text, truncation=False)
            target_tokens = self.tokenizer.encode(target_text, truncation=False)
            max_source_length = max(max_source_length, len(source_tokens))
            max_target_length = max(max_target_length, len(target_tokens))
        return max_source_length, max_target_length

def custom_collate_fn(batch):
    source_input_ids = torch.stack([item[0] for item in batch], dim=0)
    source_attention_mask = torch.stack([item[1] for item in batch], dim=0)
    source_ne_tags = torch.stack([item[2] for item in batch], dim=0)
    target_input_ids = torch.stack([item[3] for item in batch], dim=0)
    target_attention_mask = torch.stack([item[4] for item in batch], dim=0)
    return source_input_ids, source_attention_mask, source_ne_tags, target_input_ids, target_attention_mask

def create_dataloaders(data_dir, tokenizer, batch_size, num_workers=4):
    train_dataset = MT5Dataset(
        data_dir=os.path.join(data_dir, 'train'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )
    val_dataset = MT5Dataset(
        data_dir=os.path.join(data_dir, 'val'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )
    test_dataset = MT5Dataset(
        data_dir=os.path.join(data_dir, 'test'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the file paths and dataset directory structure.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Please check the file paths and dataset directory structure.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Please check the file paths and dataset directory structure.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-small")
    data_dir = "./data"
    batch_size = 1

 
    train_dataset = MT5Dataset(
        data_dir=os.path.join(data_dir, 'train'),
        source_ext='_en.txt',
        target_ext='_pt.txt',
        tokenizer=tokenizer
    )

  
    max_source_length, max_target_length = train_dataset.get_max_lengths()
    print(f"Maximum source length (in tokens): {max_source_length}")
    print(f"Maximum target length (in tokens): {max_target_length}")

    
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(data_dir, tokenizer, batch_size)
