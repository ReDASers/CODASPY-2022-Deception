"""
Reads the Amazon Scams dataset and converts it into the 

The Amazon scams dataset can be downloaded from https://github.com/aayush210789/Deception-Detection-on-Amazon-reviews-dataset
"""
import pandas as pd
import jsonlines
from langdetect import detect as detect_language
from tqdm import tqdm

RAW = 'Raw/Amazon Reviews/amazon_reviews.txt'
FAKE_LABEL = '__label1__'
OUTPUT_PATH = "amazon.jsonl"

def main():
    review_dataframe = pd.read_csv(RAW, sep='\t')
    jsonwriter = jsonlines.open(OUTPUT_PATH, 'w')
    for _, review in tqdm(review_dataframe.iterrows(), "Processing Dataset", total=len(review_dataframe)):
        text = review["REVIEW_TEXT"]
        if(detect_language(text) == "en"):
            item = {
                'text': text,
                'is_deceptive': review['LABEL'] == FAKE_LABEL
            }
            jsonwriter.write(item)

    jsonwriter.close()

if __name__ == "__main__":
    main()