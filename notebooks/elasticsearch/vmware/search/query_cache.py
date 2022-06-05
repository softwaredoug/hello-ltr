import json
import random


def essential_fields(hits):
    """
    Return a list of essential fields from a list of hits.
    """
    essential_source_fields = ['id', 'first_line', 'raw_text', 'splainer']
    for hit in hits:
        del_keys = []
        for field in hit['_source'].keys():
            if field not in essential_source_fields:
                del_keys.append(field)
        for key in del_keys:
            del hit['_source'][key]

    essential_main_fields = ['_id', '_score', '_source']
    for hit in hits:
        del_keys = []
        for field in hit.keys():
            if field not in essential_main_fields:
                del_keys.append(field)
        for key in del_keys:
            del hit[key]
    return hits


class MemoizeQuery:
    """ Adapted from
        https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python"""
    def __init__(self, f):
        self.f = f
        self.memo = {}
        self.cache_file_name = 'data/.' + f.__name__ + '.cache.jsonl'
        self.cache_file = None
        self.__name__ = f.__name__

    def __call__(self, *args, **kwargs):
        if self.cache_file is None:
            # Load cache on first call
            try:
                with open(self.cache_file_name, 'rt') as cache_file:
                    for line in cache_file:
                        cache_line = json.loads(line)
                        self.memo[cache_line['query']] = cache_line['results']
            except FileNotFoundError:
                pass
            print(f"Cache {self.cache_file_name} loaded with {len(self.memo)} entries")
            self.cache_file = open(self.cache_file_name, 'wt')
        try:
            query = kwargs['query']
            should_check = random.random() < 0.01
            if should_check and query in self.memo:
                confirm_results = essential_fields(self.f(*args, **kwargs))
                results_by_id = [r['_id'] for r in confirm_results]
                cached_by_id = [r['_id'] for r in self.memo[query]]
                assert results_by_id == cached_by_id
            elif query not in self.memo:
                self.memo[query] = essential_fields(self.f(*args, **kwargs))
            cache_line = {'query': query, 'results': essential_fields(self.memo[query])}
            self.cache_file.write(json.dumps(cache_line) + '\n')

        except KeyError:
            raise ValueError('Must pass query as kwarg to MemoizeQuery')
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[query]
