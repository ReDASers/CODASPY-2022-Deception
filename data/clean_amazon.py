import re
import html
from sys import platform

import pandas as pd

INPUT_FILE = 'Processed/amazon.jsonl'
OUTPUT_FILE = 'Cleaned/amazon.jsonl'

def clean(text):
    text = text.replace("<br />", "\n") # 3354 items
    # Unescape ASCII (1066 items)
    text = html.unescape(text)
    # Remove Amazon metadata tags 226 items
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub('[\n\r]+','\n', text)

    return text.strip()

def main():
    if platform == 'linux':
        from pandarallel import pandarallel
        pandarallel.initialize(progress_bar=False)
    
    df = pd.read_json(INPUT_FILE, orient='records', lines=True)
    print(f"Read {len(df)} samples")

    if platform == 'linux':
        df['text'] = df['text'].parallel_map(clean)
    else:
        df['text'] = df['text'].map(clean)

    # Remove duplicates
    print(f"{sum(df.duplicated())} duplicates")
    df.drop_duplicates(inplace=True)

    # Remove empty strings
    non_empty = [bool(text) and not text.isspace() for text in df['text']]
    print(f"{len(df) - sum(non_empty)} empty texts")
    df = df[non_empty].copy()

    df.to_json(OUTPUT_FILE, orient='records', lines=True)

if __name__ == '__main__':
    main()