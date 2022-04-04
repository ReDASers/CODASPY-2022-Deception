import re
import pandas as pd
from pandarallel import pandarallel
from langdetect import detect as detect_language

pandarallel.initialize()

INPUT_FILE = 'Processed/job_scams.jsonl'
OUTPUT_FILE = 'Cleaned/job_scams.jsonl'

def is_english(text):
    try:
        return detect_language(text) == "en"
    except LangDetectException:
        return False

def clean(text):
    text = text.strip()
    
    headers = [
        "description",
        "Job Summary"
    ]
    for word in headers:
        if text.lower().startswith(word):
            text = text[len(word):].strip()
    if text.startswith(':'):
        text = text[1:].strip()
    
    return text.strip()


def main():
    df = pd.read_json(INPUT_FILE, orient='records', lines=True)
    print(f"Read {len(df)} samples")

    df['text'] = df['text'].parallel_map(clean)

    # Remove Empty texts
    print(f"{len(df[df['text'].map(len) == 0])} empty texts")
    df = df[df['text'].map(len) > 0]

    # Remove placeholder descriptions
    mask = ["lorem ipsum" not in text.lower() for text in df['text']]
    print(f"{len(df) - sum(mask)} placeholders")
    df = df[mask]

    # Remove non-English
    df['is_english'] = df['text'].parallel_map(is_english)
    print(f"{sum(df['is_english'] == False)} non-Enlgish")
    df = df[df['is_english']].copy()
    df.drop('is_english', axis=1, inplace=True)
    
    # Remove duplicates
    n_dup = sum(df.duplicated("text", keep= "first"))
    print(f"{n_dup} duplicates")
    df.drop_duplicates(inplace=True, ignore_index=True)
    
    df.to_json(OUTPUT_FILE, orient='records', lines=True)
    
    print(f"{sum(df['is_deceptive'])} deceptive")
    print(f"{len(df) - sum(df['is_deceptive'])} legitimate")

if __name__ == '__main__':
    main()
