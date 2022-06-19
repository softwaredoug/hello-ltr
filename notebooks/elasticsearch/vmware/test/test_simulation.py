import pandas as pd
import math
import pytest
from simulate.diff_simulation import create_results_diff, estimate_relevance


def dcg_weight_at(rank):
    return 1 / (math.log(rank + 1, 2))


@pytest.fixture
def best_case_diff():
    results_before = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "1234"},  # relevance = 0 * 1
        {'QueryId': 1, 'DocumentId': "5678"}   # relevance = 1 *
    ])

    results_after = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "5678"},
        {'QueryId': 1, 'DocumentId': "1234"}
    ])

    diff = create_results_diff(results_before, results_after)
    actual_dcg_delta = dcg_weight_at(1) - dcg_weight_at(2)
    return diff, actual_dcg_delta


def test_best_case_prob_not_random_1(best_case_diff):
    diff, actual_dcg_delta = best_case_diff
    diff = estimate_relevance(diff, actual_dcg_delta)

    judgment_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]

    assert judgment_1234['prob_not_random'].iloc[0] == pytest.approx(1.0, 0.05), \
        "Given the diff is best case, prob_not_random should be high"
    assert judgment_5678['prob_not_random'].iloc[0] == pytest.approx(1.0, 0.05), \
        "Given the diff is best case, prob_not_random should be high"


def test_best_case_diff_(best_case_diff):
    diff, actual_dcg_delta = best_case_diff
    diff = estimate_relevance(diff, actual_dcg_delta)

    judgment_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]

    assert judgment_5678['alpha'].iloc[0] > judgment_1234['alpha'].iloc[0]
    assert judgment_5678['beta'].iloc[0] < judgment_1234['beta'].iloc[0]
    # They should be VERY different
    assert judgment_5678['alpha'].iloc[0] - judgment_1234['alpha'].iloc[0] > 0.9, \
        "Given the diff is best case, alpha diff should be very high"
    assert judgment_1234['beta'].iloc[0] - judgment_5678['beta'].iloc[0] > 0.9, \
        "Given the diff is best case, beta diff should be very high"


def test_either_swap_accounts_for_dcg_diff():
    results_before = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "1234"},
        {'QueryId': 1, 'DocumentId': "5678"},
        {'QueryId': 2, 'DocumentId': "1234"},
        {'QueryId': 2, 'DocumentId': "5678"}
    ])

    both_queries_swapped = pd.DataFrame([
        {'QueryId': 1, 'DocumentId': "5678"},
        {'QueryId': 1, 'DocumentId': "1234"},
        {'QueryId': 2, 'DocumentId': "5678"},
        {'QueryId': 2, 'DocumentId': "1234"}
    ])

    # Plausible scenarios explain this DCG change
    # Three relevant docs:
    # doc 1234 and doc 5678 are both relevant for query 1, but only 5678 is relevant for query 2
    # doc 1234 and doc 5678 are both relevant for query 2, but only 5678 is relevant for query 1
    # One relevant doc:
    # doc 1234 and doc 5678 are NOT relevant for query 1, but only 5678 is relevant for query 2
    # doc 1234 and doc 5678 are NOT relevant for query 2, but only 5678 is relevant for query 1
    #
    # To capture all of these, many simulations (total_universe_prob) need to be run to see this
    #
    diff = create_results_diff(results_before, both_queries_swapped)
    actual_dcg_delta = dcg_weight_at(1) - dcg_weight_at(2)
    diff = estimate_relevance(diff,
                              actual_dcg_delta,
                              verbose=True)

    judgment_1_1234 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "1234")]
    judgment_1_5678 = diff[(diff['QueryId'] == 1) & (diff['DocumentId'] == "5678")]
    judgment_2_1234 = diff[(diff['QueryId'] == 2) & (diff['DocumentId'] == "1234")]
    judgment_2_5678 = diff[(diff['QueryId'] == 2) & (diff['DocumentId'] == "5678")]

    assert judgment_1_5678['alpha'].iloc[0] > judgment_1_5678['beta'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"
    assert judgment_1_1234['beta'].iloc[0] > judgment_1_1234['alpha'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"
    assert judgment_2_5678['alpha'].iloc[0] > judgment_2_5678['beta'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"
    assert judgment_2_1234['beta'].iloc[0] > judgment_2_1234['alpha'].iloc[0] + 0.1, \
        "Swapping either query results in the DCG diff, so 5678s should be more relevant"

    assert judgment_1_5678['alpha'].iloc[0] == pytest.approx(judgment_2_5678['alpha'].iloc[0], 0.1), \
        "Both 5678 swaps could account for DCG delta, so should be approximately equal"
    assert judgment_1_5678['beta'].iloc[0] == pytest.approx(judgment_2_5678['beta'].iloc[0], 0.1), \
        "Both 5678 swaps could account for DCG delta, so should be approximately equal"
