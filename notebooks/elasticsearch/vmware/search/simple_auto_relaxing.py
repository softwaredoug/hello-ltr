"""A lame attempt at a simple auto-relaxing search algorithm."""

from freq_per_term import freq_per_term
import json


def simple_auto_relaxing(es, query):
    """Drop most frequent term until we get result - results were not great."""
    hits = []
    freq_terms = freq_per_term(query)
    while len(query.strip()) > 0:
        body = {
            'size': 5,
            'query': {
                'bool': {'should': [
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
