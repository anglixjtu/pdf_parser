from tqdm.auto import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    MBartForConditionalGeneration,
)
import torch


class PretrainedModel(object):
    def __init__(self, method, model_dir, dataloader, tokenizer,
                 max_target_length=512):
        self.use_gpu = torch.cuda.is_available()
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.max_target_len = max_target_length
        self.dataloader = dataloader

        if method in ['t5']:
            self.trained_model =\
                T5ForConditionalGeneration.from_pretrained(self.model_dir)
        elif method in ['mbart']:
            self.trained_model =\
                MBartForConditionalGeneration.from_pretrained(self.model_dir)

    def inference(self):
        inputs = []
        # targets = []
        outputs = []
        self.trained_model.eval()
        for batch in tqdm(self.dataloader):
            input_ids = batch['source_ids']
            input_mask = batch['source_mask']
            if self.use_gpu:
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                self.trained_model.cuda()

            output = self.trained_model.generate(input_ids=input_ids,
                                                 attention_mask=input_mask,
                                                 max_length=self.max_target_len,
                                                 temperature=1.0,
                                                 repetition_penalty=1.5)

            output_text = [
                self.tokenizer.decode(ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
                for ids in output]
            '''target_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for ids in batch["target_ids"]]'''
            input_text = [
                self.tokenizer.decode(ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
                for ids in input_ids]

            inputs.extend(input_text)
            outputs.extend(output_text)
            # targets.extend(target_text)

        return inputs, outputs
