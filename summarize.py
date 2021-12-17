import argparse
import os

from src.data import create_dataloader
from src.models import PretrainedModel
from src.utils.writer import write_to_csv


def parse_args():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                        dest='method',
                        default='mbart',
                        help='which model? t5 or mbart')
    parser.add_argument('--tknz_model_dir', type=str,
                        dest='tknz_model_dir',
                        # default='sonoisa/t5-base-japanese')
                        default='facebook/mbart-large-cc25')
    parser.add_argument('--model_dir', type=str,
                        dest='model_dir',
                        # default='sonoisa/t5-base-japanese-title-generation')
                        default='facebook/mbart-large-cc25')
    parser.add_argument('--is_fast', action='store_false',
                        dest='is_fast',
                        help='whether to use fast tokenizer or not')
    parser.add_argument('--model_base', type=str,
                        dest='model_base',
                        default='sonoisa/t5-base-japanese')
    parser.add_argument('--input_file', type=str, dest='input_file',
                        default='output/test_case_generation/manuals'
                                '/NSZN-Z68T.csv',
                        help='the cvs file for input sequences')
    parser.add_argument('--output_file', type=str, dest='output_file',
                        default='output/test_case_generation/manuals'
                                '/NSZN-Z68T_summarization.csv',
                        help='the cvs file for input sequences')
    parser.add_argument('--input_max_len', type=int, dest='input_max_len',
                        default=512,
                        help='the max length of the input sequence')
    parser.add_argument('--target_max_len', type=int, dest='target_max_len',
                        default=512,
                        help='the max length of the target sequence')
    parser.add_argument('--use_keyword', action='store_true',
                        dest='use_keyword',
                        help='whether to add summarize key words or not')
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        default=8)
    parser.add_argument('--num_workers', type=int, dest='num_workers',
                        default=4)

    args = parser.parse_args()
    return args


def run_summarization(opt):

    # create dataloaders and datasets
    test_pages = range(12, 31, 1)
    dataloader, dataset, tokenizer = \
        create_dataloader(tokenizer_name=opt.method,
                          input_file=opt.input_file,
                          split_pages=test_pages,
                          input_max_len=opt.input_max_len,
                          target_max_len=opt.target_max_len,
                          use_keyword=opt.use_keyword,
                          batch_size=opt.batch_size,
                          num_workers=opt.num_workers,
                          tknz_model_dir=opt.tknz_model_dir,
                          is_fast=opt.is_fast)

    # create model and perform inference
    pretrained_model = PretrainedModel(method=opt.method,
                                       model_dir=opt.model_dir,
                                       dataloader=dataloader,
                                       tokenizer=tokenizer,
                                       max_target_length=opt.target_max_len)
    inputs, outputs = pretrained_model.inference()

    # write to output file
    # output_file = opt.output_filename + opt.method + '.csv'
    output_dic = {opt.method+'_input': inputs, opt.method: outputs}
    write_to_csv(opt.output_file, output_dic)

    print('Finished!')


if __name__ == '__main__':
    opt = parse_args()
    run_summarization(opt)
