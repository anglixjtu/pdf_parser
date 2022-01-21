## ===================================================================================
##  Adapted from Sonoisa's T5 Japanese script
## https://github.com/sonoisa/t5-japanese/blob/main/t5_japanese_title_generation.ipynb
## ===================================================================================


from tqdm.auto import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    MBartTokenizerFast,
    MBartTokenizer,
    MBartForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch
import logging
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data.page_data import PageData
from src.data.data_collator import (FlaxDataCollatorForT5MLM,
                                    compute_input_and_target_lengths)
from src.utils.utils import decode_batch_ids

import pandas as pd

class Seq2SeqModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

        if self.hparams.method in ['t5']:
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tknz_model_dir, is_fast=self.hparams.is_fast)
            self.model =\
                T5ForConditionalGeneration.from_pretrained(
                    self.hparams.model_dir)
        elif self.hparams.method in ['mbart']:
            if self.hparams.is_fast:
                self.tokenizer = MBartTokenizerFast.from_pretrained(
                    self.hparams.tknz_model_dir)
            else:
                self.tokenizer = MBartTokenizer.from_pretrained(
                    self.hparams.tknz_model_dir)
            self.model =\
                MBartForConditionalGeneration.from_pretrained(
                    self.hparams.model_dir)

        self.output_df = {'epoch':[], 'target':[], 'output':[]}

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        """順伝搬"""
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        """ロス計算"""
        labels = batch["target_ids"]
        batch_ids=batch["source_ids"]
        masked_labels = None
        masked_input_ids = None
        
        
        if self.hparams.pretrain:
            #  Modification for pretraining by Ang Li
            noise_density = self.hparams.noise_density
            mean_noise_span_length = self.hparams.mean_noise_span_length             
            expaned_input_length, target_length =\
                compute_input_and_target_lengths(self.hparams.input_max_len, noise_density,
                                                 mean_noise_span_length) #input_ids.shape[-1]
            input_length = min(self.hparams.input_max_len, self.tokenizer.model_max_length)
            

            data_collator = FlaxDataCollatorForT5MLM(tokenizer=self.tokenizer,   
                                                     noise_density=noise_density,
                                                     mean_noise_span_length=mean_noise_span_length,
                                                     input_length=input_length,
                                                     target_length=target_length)

            masked_input_ids, masked_labels = data_collator(batch_ids, labels)
            
            pad_pos = masked_labels[:, :] == self.tokenizer.pad_token_id
            masked_labels[pad_pos] = -100
            outputs = self(input_ids=masked_input_ids,
                           labels=masked_labels)
        else:
            # All labels set to -100 are ignored (masked),
            # the loss is only computed for labels in [0, ..., config.vocab_size]
            pad_pos = labels[:, :] == self.tokenizer.pad_token_id
            labels[pad_pos] = -100 

            outputs = self(input_ids=batch_ids,
                           attention_mask=batch["source_mask"],
                           decoder_attention_mask=batch['target_mask'],
                           labels=labels)  # forward

        loss = outputs[0]

        return loss, batch_ids, labels, masked_input_ids, masked_labels

    def training_step(self, batch, batch_idx):
        """訓練ステップ処理"""
        loss, input_ids, _, masked_input_ids, masked_labels =\
             self._step(batch)
        '''output_ids = self.model.generate(masked_input_ids)

        # log the results
        init_text = decode_batch_ids(self.tokenizer, input_ids)
        input_text = decode_batch_ids(self.tokenizer, masked_input_ids)
        labels_text = decode_batch_ids(self.tokenizer, masked_labels)
        output_text = decode_batch_ids(self.tokenizer, output_ids)'''
        self.log("train_loss", loss, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """バリデーションステップ処理"""
        loss, input_ids, _, masked_input_ids, masked_labels =\
             self._step(batch)
        output_ids = self.model.generate(masked_input_ids)

        # log the results
        init_text = decode_batch_ids(self.tokenizer, input_ids)
        input_text = decode_batch_ids(self.tokenizer, masked_input_ids)
        labels_text = decode_batch_ids(self.tokenizer, masked_labels)
        output_text = decode_batch_ids(self.tokenizer, output_ids)

        self.output_df['epoch'] += [self.current_epoch]
        self.output_df['target'] += [labels_text[0]]
        self.output_df['output'] += [output_text[0]]

        output_df = pd.DataFrame(self.output_df)
        output_df.to_csv(self.hparams.out_dir+
                         "/outputs/mlm_seq.csv")

        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        loss = self._step(batch)
        self.log("test_loss", loss)
        return {"test_loss": loss}

    def configure_optimizers(self):
        """オプティマイザーとスケジューラーを作成する"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.t_total)
        self.scheduler = scheduler

        # return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        return dict(lr_scheduler=dict(scheduler=scheduler, interval='step'),
                    optimizer=optimizer)

    def get_dataset(self, tokenizer, split_pages, args):
        """データセットを作成する"""
        return PageData(tokenizer=tokenizer,
                        input_file=args.input_file,
                        target_file=args.target_file,
                        split_pages=split_pages,
                        input_max_len=args.input_max_len,
                        target_max_len=args.target_max_len,
                        use_keyword=args.use_keyword,
                        mlm=self.hparams.pretrain)

    def setup(self, stage=None):
        """初期設定（データセットの読み込み）"""
        if stage == 'fit' or stage is None:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer,
                                             split_pages=self.hparams.split_pages['train'],
                                             args=self.hparams)
            self.train_dataset = train_dataset

            val_dataset = self.get_dataset(tokenizer=self.tokenizer,
                                           split_pages=self.hparams.split_pages['valid'],
                                           args=self.hparams)
            self.val_dataset = val_dataset

            self.t_total = (
                (len(train_dataset) //
                 (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
            )

    def train_dataloader(self):
        """訓練データローダーを作成する"""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.train_batch_size,
                          drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """バリデーションデータローダーを作成する"""
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.eval_batch_size,
                          num_workers=4)
