import re
import html
from sys import platform

import pandas as pd

INPUT_FILE = 'Processed/liar_plus.jsonl'
OUTPUT_FILE = 'Cleaned/liar.jsonl'

def clean(text):
    text = text.strip()
    if text.lower().startswith("says that"): # 141 items
        text = text[9:].strip()
    if text.lower().startswith("says"): # 2685 items
        text = text[4:].strip()
    text = text.replace('"', '') # 523 items
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