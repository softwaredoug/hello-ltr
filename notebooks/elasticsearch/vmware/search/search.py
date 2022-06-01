from sys import argv
from elasticsearch import Elasticsearch

import pandas as pd
from .rerank_simple_slop_search import \
    rerank_slop_search_remaining_lines_max_snippet_at_5


def search(query,
           strategy=rerank_slop_search_remaining_lines_max_snippet_at_5):
    print(query)
    es = Elasticsearch()
    hits = strategy(es, query)
    for hit in hits:
        print("**********************************")
        print(hit['_source']['title'] if 'title' in hit['_source'] else '',
              '||',
              hit['_source']['first_line'])
        print(f"MAX SIM {hit['_source']['max_sim']} | SCORE {hit['_score']}")
        print("----------------------------------")


def submission(strategy=rerank_slop_search_remaining_lines_max_snippet_at_5,
               verbose=False):
    """Search all test queries to generate a submission."""
    queries = pd.read_csv('data/test.csv')
    all_results = []
    es = Elasticsearch()
    for query in queries.to_dict(orient='records'):
        results = strategy(es, query['Query'])
        for rank, result in enumerate(results):
            source = result['_source']
            if verbose and rank == 0:
                print(f"First result for {query['QueryId']},{query['Query']}")
                if 'titleTag' in source:
                    print(source['titleTag'])
                else:
                    print("No title tag")
                if 'first_line' in source:
                    print(source['first_line'])
                else:
                    print("No first_line")

                if 'remaining_lines' in source:
                    print(source['remaining_lines'])

            source['rank'] = rank
            source['score'] = result['_score']
            source['DocumentId'] = source['id']
            source['QueryId'] = query['QueryId']
            all_results.append(source)
    all_results = pd.DataFrame(all_results)
    return queries.merge(all_results, how='left', on='QueryId')\
        .sort_values(['QueryId', 'rank'])


def write_submission(all_results, name):
    from time import time
    timestamp = str(time()).replace('.', '')
    fname = f'data/{name}_turnbull_{timestamp}.csv'
    print("Writing To: ", fname)
    all_results[['QueryId', 'DocumentId']].to_csv(fname, index=False)


if __name__ == "__main__":
    search(argv[1])
