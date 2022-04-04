import pandas as pd
from sys import platform

import re
from langdetect import detect as detect_language, LangDetectException

URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

INPUT_FILE = 'Processed/welfake_flipped.jsonl'
OUTPUT_FILE = 'Cleaned/welfake.jsonl'

def is_english(text):
    try:
        return detect_language(text) == "en"
    except LangDetectException:
        return False

def clean(text):
    # Remove "Via: [source] from end
    if len(text) < 200:
        text = text.split("Via:")[0]
    else:
        text = text[:-200] + text[-200:].split("Via:")[0] 
    
    # Remove tags in the middle of the data
    blacklist = [
        'Via: The Nation', # 3
        'Via: Democracy Now', # 3
        'Via: Forty Six News', # 3
        'Via: Gateway Pundit', # 102
        'Via: Daily Caller', # 55
        'Via: RT', # 12
        'Via: Breitbart News', # 20
        'Via: UK Daily Mail', # 4
        'Via: CNN', # 2
        'Via: The Globe and Mail', # 1
        'Via: Guns.com', # 1
        'Via: mediaite', # 2
        'Via: Independent UK', # 1
        'Via: Independent Journal', # 1
        'Via: Independent', # 1
        'Via: The Hill', # 4
        'Via: FOX News', # 3
        'Via: NYP', # 4
        'Via: Der Tagesspeigel', # 1
        'Via: WSJ', # 1
        'Via: Scout Breitbart New', # 1
        'Via: Dallas Morning News', # 1
        'Via: Conservative Review', # 1
        
    ]
    for word in blacklist:
        text = text.replace(word, ' ')
    
    # Remove Footers
    footers = [
        'Read More:', # 69 Rows
        'Read more:', # 1464 Rows
        'FOR ENTIRE ARTICLE CLICK LINK', # 76 Rows
        'For entire story:', # 373 Rows
        'Featured image via', # 6016 Rows
        '>25% © 2016 Infowars.com', # 19 Rows
        '0% © 2016 Infowars.com', # 1 Row
        'Learn More:' # 6 Rows
        'The Daily Show with Trevor Noah Get More' # 7
    ]
    for footer in footers:
        if footer in text:
            text = text.split(footer)[0]
    
    # Remove headers
    headers = [
        '(Reuters) -' # 21256 Rows
    ]
    for header in headers:
        if header in text:
            text = text.split(header)[1]
    
    if re.search('\d+ Comments', text):
        text = re.sub('.*?\d+ Comments', '', text, count=1) # 336 Rows
        
    dashes = [
        '—', # 1881
        '--', # 43
        '-' # 103
    ]
    
    for dash in dashes:
        regex = '[\w /]{0,20} ' + dash
        # Remove [City] —
        if re.match(regex, text): 
            text = text.split(dash, 1)[1]       
    
        
    text = re.sub('\[ad3media campaign= \d+ \]', '', text)
    
    
    captions = [
        'Featured image via screenshot' # 281 Rows
    ]
    for caption in captions:
        text = text.replace(caption,'')
    
    
    re.sub(URL_REGEX, " ", text)
    
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

    # Remove non-english samples
    old_len = len(df)
    df['is_english'] = df['text'].parallel_map(is_english)
    df = df[df['is_english']].copy()
    df.drop('is_english', axis=1, inplace=True)
    print(f"{old_len - len(df)} non-Enlgish")

    df.to_json(OUTPUT_FILE, orient='records', lines=True)

if __name__ == '__main__':
    main()