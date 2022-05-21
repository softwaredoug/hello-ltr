from time import perf_counter
import json
import ast
from ltr.client import ElasticClient
from ltr.index import rebuild
import tensorflow_text
import tensorflow_hub as hub

import pandas as pd


def corpus():
    corpus = pd.read_csv('data/vmware_ir_content.csv')
    corpus = corpus.fillna('')

    parsed_rows = []

    start_time = perf_counter()

    for idx, row in enumerate(corpus.to_dict(orient='records')):
        row_dict = ast.literal_eval(row['raw_meta'])
        row_dict['id'] = row['f_name']
        row_dict['f_name'] = row['f_name']
        row_dict['raw_text'] = row['raw_text']
        row_dict['document_group'] = row['document_group']

        # fields that dont index for some reason
        bad_fields = ["DC.Date", "DC.Publisher"]
        for bad_field in bad_fields:
            if bad_field in row_dict:
                del row_dict[bad_field]

        if idx % 1000 == 0:
            print(f"Indexed {idx} documents -- {perf_counter() - start_time}")

        yield row_dict


use_mapping = {
  "properties": {
    "raw_text_use": {
        "type": "dense_vector",
        "dims": 512
    }
  }
}

use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

def process_use(doc_source):
    doc_source['raw_text_use'] = use(doc_source['raw_text']).numpy().tolist()[0]
    return doc_source




def main(version):
    client=ElasticClient()
    if version == 0:
        rebuild(client, index='vmware',
                doc_src=corpus(), force=True)
    elif version == 1:
        client.enrich(index='vmware',
                      enrich_fn=process_use,
                      mapping=use_mapping, version=version)


if __name__ == "__main__":
    from sys import argv
    version = int(argv[1])
    if version == 0:
        confirmation = input("Warning: This will delete the index and rebuild it. Continue? (y/n)")
        if confirmation == 'y':
            main(version)
    else:
        main(version)
