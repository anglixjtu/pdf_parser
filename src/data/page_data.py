import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessor.normalization import normalize_text


class PageData(Dataset):
    def __init__(self, tokenizer, input_file, split_pages,
                 input_max_len=512, target_max_len=512,
                 use_keyword=False):
        self.input_file = input_file
        self.split_pages = split_pages

        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.use_keyword = use_keyword

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, body, target):
        input = f"{body}"
        target = f"{target}"
        return input, target

    def _process_pages(self, input_df):
        """
        For each pages, concatenate paragraphs and add seperators.
        Text normalization is performed after concatenation.
        by Ang Li
        """
        if self.use_keyword:
            page_seqs = ['summarize: '] * len(self.split_pages)
        else:
            page_seqs = [''] * len(self.split_pages)
        i = 0
        concated_page = -1
        for page, para_seq in zip(input_df.page, input_df.sequence):
            # page_found = self.split_pages.index(page)
            # if page_found > -1:
            if page in self.split_pages:
                if page != concated_page and\
                        concated_page > -1:
                    page_seqs[i] = normalize_text(page_seqs[i])
                    i += 1

                page_seqs[i] += para_seq + '。'
                concated_page = page

        return page_seqs

    def _build(self):
        """
        Generate input and target tokens.
        """
        input_df = pd.read_csv(self.input_file)
        page_seqs = self._process_pages(input_df)

        for page_seq in page_seqs:
            test_cases = ''

            input, target = self._make_record(page_seq, test_cases)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input], max_length=self.input_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.target_max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
