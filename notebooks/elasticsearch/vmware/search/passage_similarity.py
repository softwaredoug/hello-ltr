import tensorflow_text   # noqa: F401
import tensorflow_hub as hub

from numpy import dot
from numpy.linalg import norm

use_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
_use = hub.load(use_path)


def use_rescore_query(es_body, rescore_depth, query, vector_field='raw_text_use'):
    query_vector = _use(query)
    es_body['rescore'] = {
        "window_size": rescore_depth,
        "query": {
          "rescore_query": {
            "script_score": {
              "query": {
                "match_all": {}
              },
              "script": {
                "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                "params": {"query_vector": query_vector}
              }
            }
          }
        }
      }
    return es_body


def passage_similarity_long_lines(query, hit,
                                  verbose=False,
                                  remaining_lines=True):
    source = hit['_source']
    vectors = [source['first_line_use']]
    for idx in range(0, 10):
        next_vector_field = f"long_remaining_lines_use_{idx}"
        if next_vector_field in source:
            vectors.append(source[next_vector_field])
    lines = [source['first_line']]
    for line in source['remaining_lines']:
        if len(line) > 20:
            lines.append(line)
    query_use = _use(query).numpy()[0]
    max_sim = -1.0
    sum_sim = 0.0
    for line, vector in zip(lines, vectors):
        cos_sim = dot(vector, query_use)/(norm(vector)*norm(query_use))
        sum_sim += cos_sim
        max_sim = max(max_sim, cos_sim)
        num_stars = 10 * (cos_sim + 1)
        if verbose:
            print(f"{cos_sim:.2f}", "*" * int(num_stars), " " * (20 - int(num_stars)), line[:40])
        if not remaining_lines:
            break
    if verbose:
        print(f"MAX: {max_sim:.2f} | SUM: {sum_sim:.2f} | SCORE: {hit['_score']}")
    return max_sim, sum_sim
