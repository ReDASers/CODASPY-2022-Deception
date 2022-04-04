"""
Reads in the Email Benchmarking Dataset using PhishBench 2.0 and converts it
to the standard format. 

The Email Benchmarking Dataset is available on request from the ReDAS lab. 
"""
import sys
import os 
import jsonlines

from phishbench.input.email_input import read_dataset_email
from langdetect import detect as detect_language, LangDetectException
from tqdm import tqdm

OUTPUT_PATH = "email_benchmarking.jsonl"

def process_email(email):
    """
    Extracts the body texts from a list of PhishBench EmailMessage objects
    """
    if email.body.text is None:
        return None
    text = email.body.text
    text = text.strip()
    if len(text) == 0:
        return None
    try:
        if detect_language(text) == "en":
            return text
    except LangDetectException:
        return None
    return None

def main(dataset_path):
    phish_path = os.path.join(dataset_path, "Phishing")
    legit_path = os.path.join(dataset_path, "Legit")
    
    emails = read_dataset_email(phish_path)[0]
    phish_emails = filter(None, map(process_email, tqdm(emails)))
    emails = read_dataset_email(legit_path)[0]
    legit_emails = filter(None, map(process_email, tqdm(emails)))

    with jsonlines.open(OUTPUT_PATH, 'w') as jsonwriter:
        for text in phish_emails:
            item = {
                # There seems to be some random quotation marks in the liar plus data. 
                'text': text,
                'is_deceptive': True
            }
            jsonwriter.write(item)
        for text in legit_emails:
            item = {
                'text': text,
                'is_deceptive': False
            }
            jsonwriter.write(item)


if __name__ == "__main__":
    main(sys.argv[1])
    