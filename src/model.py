import torch
import pytorch_lightning as pl
from torch import nn
from transformers import MT5ForConditionalGeneration
from transformers import MT5TokenizerFast, MT5ForConditionalGeneration
import sacrebleu
from src.mode_config import ModeConfig
from src.loss_functions import entity_aware_loss, ner_auxiliary_loss, placeholder_loss
import logging
import os
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=600, mode=0, num_ne_tags=26):
        super(MT5FineTuner, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.target_max_length = target_max_length
        self.mode = mode
        self.ne_tag_embedding = nn.Embedding(num_ne_tags, self.model.config.d_model)
        self.model.gradient_checkpointing_enable()
     

        logging.info(f"Initialized T5FineTuner with mode={mode}, learning_rate={learning_rate}, target_max_length={target_max_length}")
        logging.info(f"NE Tag Embedding Shape: {self.ne_tag_embedding.weight.shape}")

    def forward(self, source_token_ids, source_mask, target_token_ids=None, target_mask=None, ne_tag_mask=None, training=False):
        # Not using target mask for now 
        if training:
            labels = target_token_ids.clone()
            labels[target_token_ids == self.tokenizer.pad_token_id] = -100
            outputs = self.model(
                input_ids=source_token_ids,
                attention_mask=source_mask,
                labels=labels,
                return_dict=True
            )
            lm_logits = outputs.logits
            print(f"lm_logits shape: {lm_logits.shape}")
            print(f"labels shape: {labels.shape}")
            #token_embeddings = self.model.shared(input_ids)  # Assume shared embedding layer for input_ids
            #tag_embeddings = self.ne_tag_embedding(ne_tag_mask)
            #combined_embeddings = token_embeddings + tag_embeddings  # This adds both embeddings together

            if ne_tag_mask is not None:
                print(f"ne_tag_mask shape: {ne_tag_mask.shape}")
                assert ne_tag_mask.shape == labels.shape, f"ne_tag_mask shape {ne_tag_mask.shape} does not match labels shape {labels.shape}"
            assert lm_logits.shape[:2] == labels.shape, f"lm_logits shape {lm_logits.shape[:2]} does not match labels shape {labels.shape}"
            #if self.mode > 0 and ne_tag_mask is not None:
                #tag_embeddings = self.ne_tag_embedding(ne_tag_mask)
                #encoder_outputs = self.model.encoder(inputs_embeds=combined_embeddings, ...)


            if self.mode == 0:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            elif self.mode == 1:
                loss = entity_aware_loss(lm_logits, labels, ne_tag_mask, weight_factor=ModeConfig.MODE_1_WEIGHT)
            elif self.mode == 2:
                base_loss = nn.CrossEntropyLoss(ignore_index=-100)(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = placeholder_loss(base_loss, ne_tag_mask) * ModeConfig.MODE_2_WEIGHT
            else:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) * ModeConfig.MODE_0_WEIGHT
            return loss
        else:
            predicted_token_ids = self.model.generate(
                input_ids=source_token_ids,
                attention_mask=source_mask,
                max_length=self.target_max_length
            )
            return predicted_token_ids

    def training_step(self, batch, batch_idx):
        source_token_ids, source_mask, source_ne_tags, target_token_ids, target_mask = batch
        source_token_ids = source_token_ids.to(self.device)
        source_mask = source_mask.to(self.device)
        source_ne_tags = source_ne_tags.to(self.device)
        target_token_ids = target_token_ids.to(self.device)
        target_mask = target_mask.to(self.device)
        print(f"Batch {batch_idx} Detailed Shapes:")
        print(f"  Source Token IDs: {source_token_ids.shape}")
        print(f"  Source Mask: {source_mask.shape}")
        print(f"  Source NE Tags: {source_ne_tags.shape}")
        print(f"  Target Token IDs: {target_token_ids.shape}")
        print(f"  Target Mask: {target_mask.shape}")
        loss = self(
            source_token_ids,
            source_mask,
            target_token_ids=target_token_ids,
            target_mask=target_mask,
            ne_tag_mask=source_ne_tags,
            training=True
        )
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_token_ids, source_mask, source_ne_tags, target_token_ids, target_mask = batch
        val_loss = self(
            source_token_ids,
            source_mask,
            target_token_ids=target_token_ids,
            target_mask=target_mask,
            ne_tag_mask=source_ne_tags,
            training=True
        )
        pred_token_ids = self(
            source_token_ids,
            source_mask,
            training=False
        )
        
        for i in range(3):  # Log first 3 predictions
            print(f"Validation Sample {i + 1}:")
            print(f"  Prediction: {pred_texts[i]}")
            print(f"  Reference: {target_texts[i]}")

        #def test_step

        
        def filter_valid_ids(token_ids):
            return [id for id in token_ids if 0 <= id < self.tokenizer.vocab_size]
        pred_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in pred_token_ids]
        target_texts = [self.tokenizer.decode(filter_valid_ids(ids.tolist()), skip_special_tokens=True) for ids in target_token_ids]
        bleu_score = sacrebleu.corpus_bleu(pred_texts, [target_texts]).score
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_bleu', bleu_score, prog_bar=True)
        
        # Save the translation
        translations_dir = 'translations'
        os.makedirs(translations_dir, exist_ok=True)
        with open(os.path.join(translations_dir, f"val_epoch_{self.current_epoch}_translations.txt"), 'a', encoding='utf-8') as f:
            for pred, ref in zip(pred_texts, target_texts):
                f.write(f"Prediction: {pred}\nReference: {ref}\n\n")

        bleu_score = sacrebleu.corpus_bleu(pred_texts, [target_texts]).score
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_bleu', bleu_score, prog_bar=True)
        
        logging.info(f"Validation step {batch_idx}: BLEU Score = {bleu_score}")
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        logging.info(f"Optimizer configured: AdamW with learning rate = {self.learning_rate}")
        return optimizer
        #num_training_steps = len(self._train_dataloader) * Config.MAX_EPOCHS
        #num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        #scheduler = get_linear_schedule_with_warmup(
           # optimizer,
           # num_warmup_steps=num_warmup_steps,
           # num_training_steps=num_training_steps
        #)
        
        #return {
         #   'optimizer': optimizer,
          #  'lr_scheduler': {
           #     'scheduler': scheduler,
           #     'interval': 'step',  # or 'epoch'
            #    'name': 'linear_warmup'
           # }
       # }

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
