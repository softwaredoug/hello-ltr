import tensorflow_text   # noqa: F401
import tensorflow_hub as hub

from numpy import dot
from numpy.linalg import norm

use_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
_use = hub.load(use_path)


def passage_similarity(query, hit,
                       include_first_line=True,
                       encode_field='first_remaining_lines',
                       vector_field='long_remaining_lines_use_',
                       verbose=False):
    source = hit['_source']
    lines = []
    vectors = []
    if include_first_line:
        vectors.append(source['first_line_use'])
        lines.append(source['first_line'])
    if vector_field is not None:
        for idx in range(0, 10):
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

    max_sim = -1.0
    sum_sim = 0.0
    for line, vector in zip(lines, vectors):
        cos_sim = dot(vector, query_use)/(norm(vector)*norm(query_use))
        sum_sim += cos_sim
        max_sim = max(max_sim, cos_sim)
        num_stars = 10 * (cos_sim + 1)
        if verbose:
            print(f"{cos_sim:.2f}", "*" * int(num_stars),
                  " " * (20 - int(num_stars)), line[:40])

    msg = f"MAX: {max_sim:.2f} | SUM: {sum_sim:.2f} | SCORE: {hit['_score']}"
    if verbose:
        print(msg)

    return max_sim, sum_sim
