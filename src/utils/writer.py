import csv
import os
import pandas as pd


def write_to_csv(file, data_dict):
    append_mode = False
    if os.path.exists(file):
        df = pd.read_csv(file)
        if len(df) == len(data_dict[list(data_dict.keys())[0]]):
            append_mode=True

    if append_mode:
        for field in data_dict.keys():
            df[field]=data_dict[field]
        df.to_csv(file, index=False)
    else:
        df=pd.DataFrame.from_dict(data_dict)
        df.to_csv(file, index=False)
