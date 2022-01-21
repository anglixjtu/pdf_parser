import csv
import os
import pandas as pd
import random
import json


def write_to_csv(file, data_dict):
    append_mode = False
    if os.path.exists(file):
        df = pd.read_csv(file)
        if len(df) == len(data_dict[list(data_dict.keys())[0]]):
            append_mode = True

    if append_mode:
        for field in data_dict.keys():
            df[field] = data_dict[field]
        df.to_csv(file, index=False)
    else:
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(file, index=False, encoding='utf8')


def generate_split(file, page, split_p=[0.9, 0.05, 0.05]):
    split = {}
    n_train = int(len(page) * split_p[0])
    n_valid = int(len(page) * split_p[1])
    # n_test = len(page) - n_train - n_valid
    split['train'] = random.sample(page, n_train)
    split['valid'] = random.sample(
        list(set(page)-set(split['train'])), n_valid)
    split['test'] = list(set(page)-set(split['train']) -
                         set(split['valid']))
    jsObj = json.dumps(split, indent=4)  # indent参数是换行和缩进

    fileObject = open(file, 'w+')
    fileObject.write(jsObj)
    fileObject.close()
