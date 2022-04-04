"""
Reads in the Liar-Plus dataset and converts it to the standard format. 

We use the claim field for the text to classify. 

We consider a sample true if it is labeled as `true`, `mostly-true`, or `half-true`. 
If it is labeled as `false`, `pants-fire` or `barely-true`, we consider it false. 

The Liar-Plus dataset can be downloaded from https://github.com/Tariq60/LIAR-PLUS
"""
import jsonlines

TRAIN_PATH = "Raw/LIAR-PLUS-master/dataset/jsonl/train2.jsonl"
TEST_PATH = "Raw/LIAR-PLUS-master/dataset/jsonl/test2.jsonl"
VAL_PATH = "Raw/LIAR-PLUS-master/dataset/jsonl/val2.jsonl"

TRUE_LABELS = {'true', 'mostly-true', 'half-true'}
FALSE_LABELS = {'false', 'pants-fire', 'barely-true'}

OUTPUT_PATH = "Processed/liar_plus.jsonl"

def save_file(filename: str, writer: jsonlines.Writer):
    with jsonlines.open(filename) as reader:
        for obj in reader:
            # On 11/27/2021, we noticed that a lot of the sample claims start with "Says that [claim]" or "Says [claim]".
            # Here, we introduce additional cleaning to fix this. 
            claim: str = obj['claim']
            item = {
                # There seems to be some random quotation marks in the liar plus data. 
                'text': claim.strip(),
                'is_deceptive': obj['label'] in FALSE_LABELS
            }
            writer.write(item)

def main():
    jsonwriter = jsonlines.open(OUTPUT_PATH, 'w')
    save_file(TRAIN_PATH, jsonwriter)
    save_file(TEST_PATH, jsonwriter)
    save_file(VAL_PATH, jsonwriter)
    jsonwriter.close()
    
if __name__ == "__main__":
    main()