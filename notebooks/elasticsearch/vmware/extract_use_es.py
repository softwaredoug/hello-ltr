"""
This script evolves the vmware corpus starting with a reindex,
and adding non-destructive migrations as I learn more about the corpus,
and what might work
"""
from time import perf_counter
import json
import ast
from ltr.client import ElasticClient
from ltr.index import rebuild

import pandas as pd

# enrichment migrations
import use
import first_line
import remaining_lines

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



def main(version):
    client=ElasticClient()
    if version == 0:
        rebuild(client, index='vmware',
                doc_src=corpus(), force=True)
    elif version == 1:
        client.enrich(index='vmware',
                      enrich_fn=use.enrichment,
                      mapping=use.mapping,
                      version=version)
    elif version == 2:
        client.enrich(index='vmware',
                      enrich_fn=first_line.enrichment,
                      mapping=first_line.mapping,
                      version=version)
    elif version == 3:
        client.enrich(index='vmware',
                      enrich_fn=remaining_lines.enrichment,
                      mapping=remaining_lines.mapping, version=version)


if __name__ == "__main__":
    from sys import argv
    version = int(argv[1])
    if version == 0:
        confirmation = input("Warning: This will delete the index and rebuild it. Continue? (y/n)")
        if confirmation == 'y':
            main(version)
    else:
        main(version)
