import pandas as pd
from passage_similarity import passage_similarity
import json

colocs = pd.read_pickle('colocs_queries.pkl')
compounds = {}   # Compounds -> decompounded forms
decompounded = set()  # list of all forms decompounded from compounds
for row in colocs[colocs['compound_count'] > 0].to_dict(orient='records'):
    compounds[row['first_term'] + row['second_term']] = \
            row['first_term'] + " " + row['second_term']
    decompounded.add((row['first_term'], row['second_term']))


def with_compounds(es, query):
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
        if first_term in compounds:
            new_query.append(compounds[first_term])
        elif (first_term, second_term) in decompounded:
            new_query.append(first_term + second_term)
            fast_forward = True
        else:
            new_query.append(first_term)

    if last_term in compounds:
        new_query.append(compounds[last_term])
    else:
        new_query.append(last_term)

    print(new_query)

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
