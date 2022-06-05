import json
import random


class MemoizeQuery:
    """ Adapted from
        https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python"""
    def __init__(self, f):
        self.f = f
        self.memo = {}
        self.cache_file_name = 'data/.' + f.__name__ + '.cache.jsonl'
        try:
            for line in open(self.cache_file_name):
                cache_line = json.loads(line)
                self.memo[cache_line['query']] = cache_line['results']
        except FileNotFoundError:
            pass
        print(f"Cache {self.cache_file_name} loaded with {len(self.memo)} entries")
        self.cache_file = None

    def __call__(self, *args, **kwargs):
        if self.cache_file is None:
            self.cache_file = open(self.cache_file_name, 'wt')
        try:
            query = kwargs['query']
            should_check = random.random() < 0.01
            if should_check and query in self.memo:
                confirm_results = self.f(*args, **kwargs)
                results_by_id = [r['_id'] for r in confirm_results]
                cached_by_id = [r['_id'] for r in self.memo[query]]
                assert results_by_id == cached_by_id
            elif query not in self.memo:
                self.memo[query] = self.f(*args, **kwargs)
            cache_line = {'query': query, 'results': self.memo[query]}
            self.cache_file.write(json.dumps(cache_line) + '\n')

        except KeyError:
            raise ValueError('Must pass query as kwarg to MemoizeQuery')
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[query]
