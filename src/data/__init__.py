from torch.utils.data import Dataset, DataLoader
from .page_data import PageData
from transformers import (T5Tokenizer,
                          MBartTokenizer,
                          MBartTokenizerFast)


def create_dataloader(tokenizer_name, input_file, split_pages,
                      input_max_len=512, target_max_len=512,
                      use_keyword=False, batch_size=8,
                      num_workers=4,
                      tknz_model_dir=None,
                      is_fast=False):

    if tokenizer_name in ['t5']:
        tokenizer = T5Tokenizer.from_pretrained(
            tknz_model_dir, is_fast=is_fast)
    elif tokenizer_name in ['mbart']:
        if is_fast:
            tokenizer = MBartTokenizerFast.from_pretrained(tknz_model_dir)
        else:
            tokenizer = MBartTokenizer.from_pretrained(
            tknz_model_dir)

    dataset = PageData(tokenizer=tokenizer,
                       input_file=input_file,
                       split_pages=split_pages,
                       input_max_len=input_max_len,
                       target_max_len=target_max_len,
                       use_keyword=use_keyword)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader, dataset, tokenizer
