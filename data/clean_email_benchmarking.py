import pandas as pd
from sys import platform

import html

import re
from langdetect import detect as detect_language, LangDetectException

TAG_REGEX = re.compile("(<.*?>)+", re.DOTALL)
URL_REGEX = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

INPUT_FILE = 'Processed/email_benchmarking.jsonl'
OUTPUT_FILE = 'Cleaned/email_benchmarking.jsonl'

def is_english(text):
    try:
        return detect_language(text) == "en"
    except LangDetectException:
        return False

def clean(text):
    text = TAG_REGEX.sub('\n', text) # 5969 items
    text = re.sub("\[mailto:.*?\]", "", text) # 773 items
    text = re.sub("\[cid:.*?\]", "", text) # 299 items
    text = re.sub('\s*\n\s*','\n', text)  
    text = html.unescape(text)
    
    # Only get text part
    if m := re.search("boundary=\"(.*)\"", text):
        boundary = m.groups(1)[0]
        groups = text.split(boundary)

        if len(groups) >= 3:
            text = groups[2].strip()
        else:
            text = groups[-1].strip()
    
    # Remove Attachments
    if m := re.search("(----boundary-.*?)\n", text):
        boundary = m.groups()[0]
        groups = text.split(boundary)
        text = ""
        for group in groups:
            if "Content-Disposition: attachment;" in group:
                continue
            text += group

    if text.startswith('Content-Type'):
        text = text[text.index("\n"):].strip()
            
    text = text.replace("[IMAGE]", "").strip()  
    text = URL_REGEX.sub("", text)
    
    if m := re.search("0x\d+", text):
        text = text[0:m.start()].strip()
    
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

    # Remove non-english samples
    old_len = len(df)
    df['is_english'] = df['text'].parallel_map(is_english)
    df = df[df['is_english']].copy()
    df.drop('is_english', axis=1, inplace=True)
    print(f"{old_len - len(df)} non-Enlgish")

    df.to_json(OUTPUT_FILE, orient='records', lines=True)

if __name__ == '__main__':
    main()