from collections import Counter, defaultdict
import string
import pandas as pd
import elasticsearch.helpers
from elasticsearch import Elasticsearch
from time import perf_counter

class Colocations:

    def __init__(self):
        self.first_terms = defaultdict(Counter)   # every term paired with other terms that occur after it
        self.snippet_counts = defaultdict(Counter)
        self.scores = defaultdict(lambda: {})   # every term paired with other terms that occur after it
        self.first_counts = Counter()  # count of when term goes first
        self.second_counts = Counter()  # count of when term goes first
        self.N = 0 # total number of pairs
        self.snippets = 0 # total number of snippets

    # See https://opensourceconnections.com/blog/2019/05/16/unreasonable-effectiveness-of-collocations/
    def _batch_score(self):
        """Chi square score each co-occurrence pair"""
        N = sum(self.first_counts.values())
        scores = []
        print(f"Scoring {len(self.first_counts)} co-occurrence pairs -- {N} total co-occurrences")
        for idx, (first_term, first_total) in enumerate(self.first_counts.items()):
            if first_total > 50:
                print(idx, first_term, first_total)
                for second_term, second_total in self.second_counts.items():
                    together = self.first_terms[first_term][second_term]
                    if together <= 10:
                        continue

                    first_without = first_total - together
                    second_without = second_total - together
                    neither = N - together - first_without - second_without

                    numerator = N * (((together * neither) - (first_without * second_without)) ** 2)

                    all_without_first = N - first_total
                    all_without_second = N - second_total

                    denominator = (first_total * second_total * all_without_first * all_without_second)

                    score = -1.0
                    if denominator != 0:
                        score = numerator / denominator

                    scores.append({'first_term': first_term,
                                   'second_term': second_term,
                                   'score': score})
        return pd.DataFrame(scores).sort_values('score', ascending=False)


    def scores_to_dataframe(self, min_count=50, min_snippet_count=70):
        scores = []
        for first, seconds in self.scores.items():
            for second, score in seconds.items():
                compound = first + second
                compound_total = self.first_counts[compound]
                count = self.first_terms[first][second]
                snippet_count = self.snippet_counts[first][second]
                if count > min_count and snippet_count > min_snippet_count:
                    scores.append({'first_term': first,
                                   'second_term': second,
                                   'score': score,
                                   'count': count,
                                   'compound_count': compound_total,
                                   'snippet_count': snippet_count})
        if len(scores) > 0:
            return pd.DataFrame(scores).sort_values('score', ascending=False)
        else:
            return pd.DataFrame([])


    def add_text(self, texts):
        """
        :param text: a string
        :return: a list of tuples of the form (term, term)
        """
        self.snippets += 1
        unique_terms = set()
        for text in texts:
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.split()

            for idx, (first, second) in enumerate(zip(text, text[1:])):
                self.N += 1
                self.first_terms[first][second] += 1
                self.first_counts[first] += 1
                self.second_counts[second] += 1


                unique_terms.add( (first,second) )

                together = self.first_terms[first][second]
                first_total = self.first_counts[first]
                second_total = self.second_counts[second]
                neither = self.N - together - first_total - second_total
                numerator = self.N * (((together * neither) - (first_total * second_total)) ** 2)

                all_without_first = self.N - first_total
                all_without_second = self.N - second_total

                denominator = (first_total * second_total * all_without_first * all_without_second)
                if denominator != 0:
                    self.scores[first][second] = numerator / denominator
                    # print(f"{first} {second} {self.scores[first][second]}")

        for first, second in unique_terms:
            self.snippet_counts[first][second] += 1



def scan_index(index, query={"_source": ["raw_text"], "query": {"match_all": {}}}):
    """Compute colocation and compound scores for the corpus."""
    es = Elasticsearch('http://localhost:9200', timeout=30, max_retries=10,
                       retry_on_status=True, retry_on_timeout=True)
    start = perf_counter()
    collocations = Colocations()
    for idx, doc in enumerate(elasticsearch.helpers.scan(es, index=index, scroll='5m',
                                                         size=1000,
                                                         query=query)):
        collocations.add_text(doc['_source']['raw_text'].split("\n")[:5])
        #collocations.add_text(doc['_source']['first_line'])
        #for line in doc['_source']['remaining_lines'][:10]:
        #    collocations.add_text(line)
        if idx % 10000 == 0:
            print(f"Scanned {idx} documents -- {perf_counter() - start}")
            print(collocations.scores_to_dataframe().head(10))
    colos = collocations.scores_to_dataframe()
    colos.to_pickle('colocs.pkl')


def scan_queries():
    """Compute colocation and compound scores for test queries."""
    start = perf_counter()
    collocations = Colocations()
    queries = pd.read_csv('data/test.csv')
    for query in queries.Query:
        collocations.add_text([query])
        #collocations.add_text(doc['_source']['first_line'])
        #for line in doc['_source']['remaining_lines'][:10]:
        #    collocations.add_text(line)
    print(collocations.scores_to_dataframe(min_count=0, min_snippet_count=0).head(10))
    colos = collocations.scores_to_dataframe(min_count=0, min_snippet_count=0)
    colos.to_pickle('colocs_queries.pkl')



if __name__ == "__main__":
    from sys import argv
    client = ElasticClient()
    scan_index(client, 'vmware')
