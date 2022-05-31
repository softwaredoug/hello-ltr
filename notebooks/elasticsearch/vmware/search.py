from ltr.client import ElasticClient
from sys import argv
import json

import tensorflow_text
import tensorflow_hub as hub

from numpy import dot
from numpy.linalg import norm
import pandas as pd

_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")


def freq_per_term(query):
    client = ElasticClient()
    terms = query.split()
    es = client.es

    bodies = [{
        "query": {"match": {
            "raw_text": term
        }}} for term in terms]

    counts = []
    for term, body in zip(terms, bodies):
        resp = es.count(index='vmware', body=body)
        counts.append((term, resp['count']))
    return dict(counts)


def submission_2_search(client, query):
    """Search for submission that got NDCG ~0.29."""
    es = client.es
    body = {
        'size': 5,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    return es.search(index='vmware', body=body)['hits']['hits']


def max_passage_rerank_at_5(client, query):
    """Rerank top 5 submissions by max passage USE similarity. NDCG of 0.30562"""
    es = client.es
    body = {
        'size': 5,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = passage_similarity(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


def sum_passage_rerank_at_5(client, query):
    """Rerank top 5 submissions by max passage USE similarity. NDCG of 0.30550"""
    es = client.es
    body = {
        'size': 5,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = passage_similarity(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['sum_sim'], reverse=True)
    hits = hits[:5]
    return hits


def max_passage_rerank_at_50(client, query):
    """Rerank top 50 submissions by max passage USE similarity"""
    es = client.es
    body = {
        'size': 50,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'raw_text': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = passage_similarity(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


def bm25_raw_text_to_remaining_lines_search(client, query):
    """Best pure BM25 submission on 29-May ~0.305."""
    es = client.es
    body = {
        'size': 5,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'remaining_lines': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    return es.search(index='vmware', body=body)['hits']['hits']


def max_passage_rerank_at_5_attempt_2(client, query):
    """Rerank with USE on top of best pure BM25 submission on 29-May NDCG 0.31569."""
    es = client.es
    body = {
        'size': 5,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'remaining_lines': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = passage_similarity(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


def max_passage_rerank_first_remaining_lines(client, query):
    """Try only the earliest remaining lines for matching (NDCG, 0.29)"""
    es = client.es
    body = {
        'size': 5,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'first_remaining_lines': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'first_remaining_lines': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = passage_similarity(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


def simple_auto_relaxing(client, query):
    """Drop most frequent term until we get result - results were not great :)"""
    es = client.es
    hits = []
    freq_terms = freq_per_term(query)
    while len(query.strip()) > 0:
        body = {
            'size': 5,
            'query': {
                'bool': { 'should': [
                    {'match_phrase': {
                        'raw_text': {
                            'slop': 50,
                            'query': query
                        }
                    }},
                    {'match_phrase': {
                        'first_line': {
                            'slop': 10,
                            'query': query
                        }
                    }},
                ]}
            }
        }

        print(json.dumps(body, indent=2))

        hits = es.search(index='vmware', body=body)['hits']['hits']

        if len(hits) > 0:
            return hits
        else:
            terms = query.split()
            least_freq_term_df = 99999999999999999
            drop_term = ''
            for term in terms:
                term_df = freq_terms[term]
                if term_df <= least_freq_term_df:
                    least_freq_term_df = term_df
                    drop_term = term
            query = query.replace(drop_term, ' ')


def rerank_document_strides(client, query):
    """Rerank with USE on top of 60 term strides"""
    es = client.es
    body = {
        'size': 20,
        'query': {
            'bool': { 'should': [
                {'match_phrase': {
                    'sixty_token_strides': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match_phrase': {
                    'first_line': {
                        'slop': 10,
                        'query': query
                    }
                }},
                {'match': {
                    'sixty_token_strides': {
                        'query': query
                    }
                }},
                {'match': {
                    'first_line': {
                        'query': query
                    }
                }},
            ]}
        }
    }

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = passage_similarity(query, hit,
                vector_field=None,
                encode_field='sixty_token_strides',
                verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits


def passage_similarity(query, hit,
                       include_first_line = True,
                       encode_field = 'first_remaining_lines',
                       vector_field='long_remaining_lines_use_', verbose=False):
    source = hit['_source']
    lines = []
    vectors = []
    if include_first_line:
        vectors.append(source['first_line_use'])
        lines.append(source['first_line'])
    if vector_field is not None:
        for idx in range(0,10):
            next_vector_field = f"{vector_field}_{idx}"
            if next_vector_field in source:
                vectors.append(source[next_vector_field])
        # TODO - fix me - hard coded!
        for line in source['remaining_lines']:
            if len(line) > 20:
                lines.append(line)
    elif encode_field is not None:
        if isinstance(source[encode_field], list):
            for text in source[encode_field]:
                vectors.append(_use(text))
                lines.append(text)
        else:
            vectors.append(_use(source[encode_field]))


    query_use = _use(query).numpy()[0]

    max_sim = -1.0; sum_sim = 0.0
    for line, vector in zip(lines, vectors):
        cos_sim = dot(vector, query_use)/(norm(vector)*norm(query_use))
        sum_sim += cos_sim
        max_sim = max(max_sim, cos_sim)
        num_stars = 10 * (cos_sim + 1)
        if verbose:
            print(f"{cos_sim:.2f}", "*" * int(num_stars), " " * (20 - int(num_stars)), line[:40])

    if verbose:
        print(f"MAX: {max_sim:.2f} | SUM: {sum_sim:.2f} | SCORE: {hit['_score']}")

    return max_sim, sum_sim



def search(query, strategy=max_passage_rerank_at_5_attempt_2):
    """Search for submission that got NDCG ~0.29."""
    client = ElasticClient()
    print(query)
    for hit in strategy(client, query):
        print("**********************************")
        print(hit['_source']['title'] if 'title' in hit['_source'] else '', '||',
              hit['_source']['first_line'])
        max_sim, sum_sim = passage_similarity(query, hit, verbose=True)
        print("----------------------------------")


def submission(strategy=max_passage_rerank_at_5, verbose=False):
    client = ElasticClient()
    queries = pd.read_csv('data/test.csv')
    all_results = []
    for query in queries.to_dict(orient='records'):
        results = strategy(client, query['Query'])
        for rank, result in enumerate(results):
            source = result['_source']
            if verbose and rank == 0:
                print(f"First result for for {query['QueryId']},{query['Query']}")
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
    return queries.merge(all_results, how='left', on='QueryId').sort_values(['QueryId', 'rank'])


def write_submission(all_results, name):
    from time import time
    timestamp = str(time()).replace('.', '')
    fname = f'data/{name}_turnbull_{timestamp}.csv'
    print("Writing To: ", fname)
    all_results[['QueryId', 'DocumentId']].to_csv(fname, index=False)


if __name__ == "__main__":
    search(argv[1])
