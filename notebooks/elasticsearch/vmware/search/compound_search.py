import pandas as pd
from .passage_similarity import passage_similarity
from . import freq_per_term, freq_per_phrase
import json


def get_compound_dicts():
    colocs = pd.read_pickle('colocs_queries.pkl')
    to_decompound = {}   # Compounds -> decompounded forms
    to_compound = set()  # list of all forms decompounded from compounds
    for row in colocs[colocs['compound_count'] > 0].to_dict(orient='records'):
        compound_freq = freq_per_term(row['first_term'] + row['second_term'])
        decompound_freq = freq_per_phrase([row['first_term'] +
                                           " " + row['second_term']])
        assert len(compound_freq.values()) == 1
        assert len(decompound_freq.values()) == 1
        compound_freq = list(compound_freq.values())[0]
        decompound_freq = list(decompound_freq.values())[0]
        print(row['first_term'] + " " + row['second_term'],
              compound_freq, decompound_freq)
        if compound_freq < decompound_freq:
            # What we want to DECOMPOUND
            to_decompound[row['first_term'] + row['second_term']] = \
                    row['first_term'] + " " + row['second_term']
        else:
            to_compound.add((row['first_term'], row['second_term']))
    return to_decompound, to_compound


to_decompound, to_compound = get_compound_dicts()


def with_compounds_at_20(es, query):
    """Adds using compounds computed from query dataset."""
    body = {
        'size': 20,
        'query': {
            'bool': {'should': [
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

    new_query = []
    last_term = ''
    fast_forward = False
    for first_term, second_term in zip(query.split(), query.split()[1:]):
        last_term = second_term
        if fast_forward:
            print("Skipping: " + first_term + " " + second_term)
            fast_forward = False
            continue
        first_term = first_term.strip().lower()
        second_term = second_term.strip().lower()
        if first_term in to_decompound:
            new_query.append(to_decompound[first_term])
        elif (first_term, second_term) in to_compound:
            new_query.append(first_term + second_term)
            fast_forward = True
        else:
            new_query.append(first_term)

    if last_term in to_decompound:
        new_query.append(to_decompound[last_term])
    else:
        new_query.append(last_term)

    if new_query != query.split():
        new_query = " ".join(new_query)
        alt_clauses = [{'match_phrase': {
            'remaining_lines': {
                'slop': 10,
                'query': new_query
            }
        }},
        {'match_phrase': {
            'first_line': {
                'slop': 10,
                'query': new_query
            }
        }},
        {'match': {
            'remaining_lines': {
                'query': new_query
            }
        }},
        {'match': {
            'first_line': {
                'query': new_query
            }
        }}]
        body['query']['bool']['should'].extend(alt_clauses)

    print(json.dumps(body, indent=2))

    hits = es.search(index='vmware', body=body)['hits']['hits']

    for hit in hits:
        hit['_source']['max_sim'], hit['_source']['sum_sim'] = \
            passage_similarity(query, hit, verbose=False)

    hits = sorted(hits, key=lambda x: x['_source']['max_sim'], reverse=True)
    hits = hits[:5]
    return hits
