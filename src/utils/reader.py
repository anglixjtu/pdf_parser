import json


def load_json_split(file, train_num=0):
    with open(file, 'r') as load_f:
        split = json.load(load_f)

    if train_num > 0:
        split['train'] = split['train'][:train_num]
    return split
