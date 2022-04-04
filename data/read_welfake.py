import csv
import sys
import pandas as pd
from tqdm import tqdm
import jsonlines

RAW_PATH = 'Raw/WELFake_Dataset.csv'

def read_welfake():
    csv.field_size_limit(123456789)
    welfake = pd.read_csv(RAW_PATH, engine='python', encoding='utf-8')

    dic_list = []
    for _, row in tqdm(welfake.iterrows(), total=72134):
        text = row['text']
        if not isinstance(text, str) or len(text) == 0 or text.isspace():
            continue
        dic_list.append({'text': text, 'is_deceptive': row['label'] == 1})

    with jsonlines.open('welfake_flipped.jsonl', 'w') as out_file:
        out_file.write_all(dic_list)

if __name__ == "__main__":
    read_welfake()