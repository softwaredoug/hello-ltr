#!pip install tensorflow
#!pip install tensorflow_hub
#!pip install tensorflow_text

#import os.path
#if not os.path.isfile('data/vmware_ir_content.csv'):
#    !pip install kaggle
#    !kaggle competitions download -c vmware-zero-shot-information-retrieval
#    !mkdir -p data/
#    !unzip -o vmware-zero-shot-information-retrieval.zip
#    !mv *.csv data/

import tensorflow_text
import tensorflow_hub as hub
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

import pandas as pd
corpus = pd.read_csv('notebooks/elasticsearch/vmware/data/vmware_ir_content.csv')
queries = pd.read_csv('notebooks/elasticsearch/vmware/data/test.csv')
corpus = corpus.fillna('')


from time import perf_counter

import json
# training/content csv data
# test csv data
import concurrent.futures
# corpus = pd.read_csv("/kaggle/input/vmware-zero-shot-information-retrieval/vmware_ir_content.csv")
# queries = pd.read_csv("/kaggle/input/vmware-zero-shot-information-retrieval/test.csv")
corpus = corpus.fillna('')

import json
import ast

counter = 0

def parse_corpus_row(row):
    """Parse the metadata and eval USE on raw text"""
    row_dict = ast.literal_eval(row['raw_meta'])
    row_dict['id'] = row['f_name']
    row_dict['f_name'] = row['f_name']
    row_dict['raw_text'] = row['raw_text']
    row_dict['document_group'] = row['document_group']
    row_dict['raw_text_use'] = use(row['raw_text'])

    # fields that dont index for some reason
    bad_fields = ["DC.Date", "DC.Publisher"]
    for bad_field in bad_fields:
        if bad_field in row_dict:
            del row_dict[bad_field]
    return row_dict

parsed_rows = []

start = perf_counter()
print_every = 100
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    print("Launching ", start)
    row_futures = [executor.submit(parse_corpus_row, row) for row in corpus.to_dict(orient='records')]
    for num, future in enumerate(concurrent.futures.as_completed(row_futures)):
        result = future.result()
        parsed_rows.append(result)
        if num % print_every == 0:
            print(num, (perf_counter() - start) / (num + 1), result['id'])

corpus = pd.DataFrame(parsed_rows)
print("Done, writing...")
corpus.to_pickle('notebooks/elasticsearch/vmware/data/vmware_ir_content_parsed.pkl')
