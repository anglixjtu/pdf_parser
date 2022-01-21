import argparse
from json import load
import os
import torch
import pytorch_lightning as pl

from src.data import create_dataloader
from src.models.seq2seq import Seq2SeqModel
from src.utils.writer import write_to_csv
from src.utils.reader import load_json_split
import pandas as pd

def parse_args():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                        dest='method',
                        default='t5',
                        help='which model? t5 or mbart')
    parser.add_argument('--run_name', type=str,
                        dest='run_name',
                        default='pretrain-ahn-128',
                        help='run name')
    parser.add_argument('--exp_name', type=str,
                        dest='exp_name',
                        default='test_cases_generation',
                        help='experiment name')
    parser.add_argument('--tknz_model_dir', type=str,
                        dest='tknz_model_dir',
                        default='sonoisa/t5-base-japanese')
    # default='facebook/mbart-large-cc25')
    parser.add_argument('--model_dir', type=str,
                        dest='model_dir',
                        default='sonoisa/t5-base-japanese')
    parser.add_argument('--out_dir', type=str,
                        dest='out_dir',
                        default='output/test_case_generation/manuals/pretrain/')
    # default='facebook/mbart-large-cc25')
    parser.add_argument('--is_fast', action='store_false',
                        dest='is_fast',
                        help='whether to use fast tokenizer or not')
    parser.add_argument('--pretrain', action='store_false',
                        dest='pretrain',
                        help='whether to pretrain or not')
    parser.add_argument('--model_base', type=str,
                        dest='model_base',
                        default='sonoisa/t5-base-japanese')
    parser.add_argument('--input_file', type=str, dest='input_file',
                        default='data/Manuals/ahn.csv',
                        help='the cvs file for input sequences')
    parser.add_argument('--target_file', type=str, dest='target_file',
                        default=None, #'data/News/open_news_target.csv'
                        help='the cvs file for target sequences')
    parser.add_argument('--split_file', type=str, dest='split_file',
                        default='data/Manuals/ahn_split.csv',
                        help='the json file for train/valid/test split')
    parser.add_argument('--output_file', type=str, dest='output_file',
                        default='output/test_case_generation/manuals/pretrain/'
                        'ahn.csv',
                        help='the cvs file for output sequences')
    parser.add_argument('--use_keyword', action='store_true',
                        dest='use_keyword',
                        help='whether to add summarize key words or not')
    # training hyper-parameters
    parser.add_argument('--train_num', type=int, dest='train_num',
                        default=0,
                        help='the number of train items. If 0, use all.')
    parser.add_argument('--input_max_len', type=int, dest='input_max_len',
                        default=512,
                        help='the max length of the input sequence')
    parser.add_argument('--target_max_len', type=int, dest='target_max_len',
                        default=128,
                        help='the max length of the target sequence')
    parser.add_argument('--train_batch_size', type=int, dest='train_batch_size',
                        default=8)
    parser.add_argument('--eval_batch_size', type=int, dest='eval_batch_size',
                        default=8)
    parser.add_argument('--num_train_epochs', type=int, dest='num_train_epochs',
                        default=40)
    parser.add_argument('--num_workers', type=int, dest='num_workers',
                        default=4)
    parser.add_argument('--learning_rate', type=float, dest='learning_rate',
                        default=3e-4)
    parser.add_argument('--weight_decay', type=float, dest='weight_decay',
                        default=0.0)
    parser.add_argument('--adam_epsilon', type=float, dest='adam_epsilon',
                        default=1e-8)
    parser.add_argument('--warmup_steps', type=float, dest='warmup_steps',
                        default=0)
    parser.add_argument('--n_gpu', type=int, dest='n_gpu',
                        default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        dest='gradient_accumulation_steps',
                        default=1)
    # pre-training hyper-parameters
    parser.add_argument('--noise_density', type=float, dest='noise_density',
                        default=0.15,
                        help='Ratio of tokens to mask for span masked language modeling loss')
    parser.add_argument('--mean_noise_span_length', type=float, dest='mean_noise_span_length',
                        default=3.0,
                        help='Mean span length of masked tokens')

    args = parser.parse_args()
    return args


def run_training(opt):
    orig_opt_dict = vars(opt).copy()

    # set hyper-parameters
    split_pages = load_json_split(opt.split_file, opt.train_num)
    opt.split_pages = split_pages

    USE_GPU = torch.cuda.is_available()

    train_params = dict(
        accumulate_grad_batches=1,
        gpus=1 if USE_GPU else 0,
        max_epochs=opt.num_train_epochs,
        precision=32,        # NaN output if fp=16
        #amp_level="apex",
        #amp_backend='apex',
        gradient_clip_val=1.0,
    )

    # Train the model
    model = Seq2SeqModel(vars(opt))
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)


    # 最終エポックのモデルを保存
    out_dir =  '%s%s_%s_%s_%s/'%(opt.out_dir,opt.name,opt.method, opt.run_name, opt.train_num)
    model.tokenizer.save_pretrained(out_dir)
    model.model.save_pretrained(out_dir)

    del model

    print('Finished!')


if __name__ == '__main__':
    opt = parse_args()
    run_training(opt)
