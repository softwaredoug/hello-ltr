from .search import search, submission, write_submission  # noqa
from elasticsearch import Elasticsearch


def freq_per_term(query):
    """Get the frequency for term via naive match query."""
    terms = query.split()
    es = Elasticsearch()

    bodies = [{
        "query": {"match": {
            "raw_text": term
        }}} for term in terms]

    counts = []
    for term, body in zip(terms, bodies):
        resp = es.count(index='vmware', body=body)
        counts.append((term, resp['count']))
    return dict(counts)


def freq_per_phrase(phrases):
    """Get the frequency for term via naive match query."""
    es = Elasticsearch()

    bodies = [{
        "query": {"match_phrase": {
            "raw_text": phrase
        }}} for phrase in phrases]

    counts = []
    for term, body in zip(phrases, bodies):
        resp = es.count(index='vmware', body=body)
        counts.append((term, resp['count']))
    return dict(counts)
