"""
Reads the job scams dataset and converts to the standard format.

Uses the job description as the text to classifiy.
Jobs are labeled deceptive if they're marked as fraudlent. 

The job scams dataset can be donwloaded from https://www.kaggle.com/amruthjithrajvr/recruitment-scam
"""

import csv
import re
from multiprocessing import Pool

import jsonlines
import langdetect
from tqdm import tqdm
from bs4 import BeautifulSoup

JOB_SCAM_PATH = "Raw/Job Scams/DataSet.csv"

TAG_REGEX = re.compile(r"\S*#[A-Z]+_[a-f0-9]*#\S*")


def parse_row(row):
    soup = BeautifulSoup(row['description'], features="html.parser")
    # Remove excess whitespace
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = re.sub(r"\s+", " ", soup.text).strip()
    # Remove HTML tags
    description = TAG_REGEX.sub("", text)

    return {
        "text": description.strip(),
        "is_deceptive": True if row['fraudulent'] == 'f' else False
    }


def is_deceptive(value: dict):
    return value["is_deceptive"]


def read_job_scams(raw_path, output_path):

    with open(raw_path, encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    with Pool() as pool:
        data = list(pool.imap(parse_row, tqdm(rows, "Reading Job Scams")))
            
    num_items = len(data)
    num_deceptive = sum(map(is_deceptive, data))
    print(f"{num_items} items loaded")
    print(f"    {num_deceptive} deceptive items")
    print(f"    {num_items - num_deceptive} legitimate items")

    with jsonlines.open(output_path, 'w') as out_file:
        out_file.write_all(data)


if __name__ == "__main__":
    read_job_scams(JOB_SCAM_PATH, "Processed/job_scams.jsonl")
