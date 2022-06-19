import pandas as pd
import numpy as np
from statistics import NormalDist
from time import perf_counter

ideal_dcg_at_5 = 2.948459119


def add_weights(diff):
    """Add a position. Compute DCG weights"""
    diff['position'] = diff.groupby('QueryId').cumcount() + 1
    diff['weight'] = 1 / np.log2(diff['position'] + 1)
    return diff


# Ideeal DCG@5 with weight 1 / log(n+1), if n is 1 based position
ideal_dcg_at_5 = 2.948459119
# one unit of NDCG == ideal_dcg units of DCG
#
# total DCG change =
#  NDCG * ideal_dcg * num_queries
#
#  0-1 total possible sales (1 highest possible)
#  ideal sales number
#
# what if this was an A/B test instead of NDCG?
#  - assume some weighted correlation of position ranking to A/B testing outcome
#  - we could somehow know those weights (ie best item in position 1 nets 5 sales. But in position 2 nets 3)
#  - given total sales between two rankings
#  -
#
# (note this -- imperfectly -- forces the NDCG from the source system into our DCG scaling)
#
# For every new result, with DCG position weight wn. Each result has gone from position wn -> wm
#
# For every moved search result with a relevance grade g, we can see the change in DCG as follows
#
#  delta_dcg = (wn - wm) * g + ...  (for every changed query/doc pair)
#
# Where we assume grade g is either 0 (irrelevant) or 1 (relevant)
#
# Perfect case, only permutation that creates this is when they're all 1
#
#  max_delta_dcg = sum(wn - wm) for all wn > wm
#
# When we detect an actuall delta_dcg, we can randomly select grades 0 and 1 to each doc to see
# which ones most approximate the delta dcg


def create_results_diff(results_before, results_after):
    """Compute the DCG delta weights resulting from the before and after,
       so we can compare to observed mean delta dcg."""

    results_before = results_before.groupby('QueryId').head(5)
    results_after = results_after.groupby('QueryId').head(5)
    assert len(results_before) == len(results_after)

    results_before = add_weights(results_before)
    results_after = add_weights(results_after)
    diff = results_before.merge(results_after,
                                on=['QueryId', 'DocumentId'],
                                how='outer')
    diff = diff.rename(
        columns={'position_x': 'position_before',
                 'position_y': 'position_after',
                 'weight_x': 'weight_before',
                 'weight_y': 'weight_after'}
    )
    # For each document, its DCG weight before and aftter
    # REVIEW FOR BUG
    diff['weight_after'] = diff['weight_after'].replace(np.nan, 0)
    diff['weight_before'] = diff['weight_before'].replace(np.nan, 0)
    diff['weight_delta'] = diff['weight_after'] - diff['weight_before']
    diff['position_delta'] = diff['position_after'] - diff['position_before']
    diff['weight_delta_abs'] = np.abs(diff['weight_delta'])
    diff.loc[:, 'alpha'] = 0.001
    diff.loc[:, 'beta'] = 0.001
    assert (diff[diff['position_delta'] == 0]['weight_delta'] == 0).all()

    diff['grade'] = 0
    diff.loc[diff['weight_delta'] > 0, 'grade'] = 1
    diff.loc[diff['weight_delta'] < 0, 'grade'] = 0
    best_case_dcg_delta = sum(diff['grade'] * diff['weight_delta'])

    diff['best_case_dcg_delta'] = best_case_dcg_delta

    return diff


def _sign(x):
    return -1.0 if x < 0.0 else 1.0


def _biased_random_sample(sample_size, prob_of_relevance=0.9):
    sample = np.random.random_sample(sample_size)
    biased_sample = np.ones(sample_size)
    biased_sample[sample >= prob_of_relevance] = 0
    return biased_sample.astype(int)


def _universe_probability(actual_dcg_delta, simulated_dcg_delta, std_dev=1):
    if abs(actual_dcg_delta - simulated_dcg_delta) > (3 * std_dev):
        return 0
    actual_universe_distribution = NormalDist(mu=actual_dcg_delta, sigma=std_dev)
    simulated_universe_distribution = NormalDist(mu=simulated_dcg_delta, sigma=std_dev)
    universe_prob = actual_universe_distribution.overlap(simulated_universe_distribution)

    return universe_prob




def _debug_query(diff, query_id):
    """Examine a single query's diff to see what it says about relevance."""
    query = diff[diff['QueryId'] == query_id]
    query = query.merge(corpus, right_on='f_name', left_on='DocumentId', how='left')

    for row in query.sort_values('position_before').to_dict('records'):
        print(row['QueryId'], row['raw_text'][:40], '|',
              row['position_before'], '->', row['position_after'], '|',
              f"{row['weight_delta']:.3f} alpha:{row['alpha']:.2f} beta:{row['beta']:.2f}")


def estimate_relevance(diff, actual_dcg_delta, min_rounds=100, converge_std_dev=0.02, verbose=False,
                       dcg_diff_std_dev=None):
    """Simulate a single ranking change and account for the actual dcg delta by guessing result relevance.

    For each oserved query/doc ranking change:
    - alpha counts simulations explaining the change when query/doc grade=1 (relevant),
    - beta when its grade=0 (not relevant).

    More specifically, alpha/beta don't _count_, they accumulate the probability the simulated universe's DCG equals the actual
    The `dcg_diff_std_dev` is the standard deviation of the actual DCG change used to calibrate how exact this should be. It defaults to
    0.01 * sqrt(number of queries).

    The algorithim iterates, incrementing alpha and beta approriately, based on the observed DCG diff of random
    relevance grades. The alphas and betas keep incrementing untilthe std dev falls below the converge_std_dev threshold.
    That is, they're not really changing much more relative to each other.

    Then the alpha/beta are scaled back to the total_universe_prob.

    Combining this diff with other diffs is left as an exercise for the caller. Simply summing can underweight very certain scenarios,
    relative to much more uncertain ones. So you may wish to consider `prob_not_random` and `total_universe_prob` to adjust
    alpha/beta before combining with other diffs.

    Returns modified diff dataframe adding columns:

        - alpha: proportion of universes observed to be relevant (grade=1) to account for actual dcg delta
        - beta:  proportion of universes observed to be irrelevant (grade=0) to account for actual dcg delta
        - prob_not_random: probability observed rank changes explains the actual DCG delta
        - std_dev: std dev of beta distribution represented by alpha and beta
        - total_universe_prob: the total probability density of universes that occured in this simulation


    """
    num_grades_changed = len(diff.loc[diff['weight_delta'] != 0])

    best_case_dcg_delta = diff.iloc[0]['best_case_dcg_delta']

    # Hacky attempt to estimate the probablity the observed DCG is not random
    num_grades_changed = len(diff.loc[diff['weight_delta'] != 0])
    likelihood_not_random = (num_grades_changed**(actual_dcg_delta / best_case_dcg_delta)) / num_grades_changed
    diff['prob_not_random'] = likelihood_not_random

    if dcg_diff_std_dev is None:
        dcg_diff_std_dev = 0.01 * np.sqrt(diff['QueryId'].nunique())

    # likelihood_not_random is 0 -> 0.5
    # likelihood_not_random is 1 -> 0.9
    prob_positive = (((likelihood_not_random / 2) + 0.5) - 0.01)
    best_universe_prob = 0.0

    plausible_universe_prob = 0

    # If the probability of observing actual DCG at random is pretty high
    # then we shouldn't increment alpha and beta too much
    # otherwise alpha and beta become too certain around
    #
    # OTOH if the probability is very unlikely, the actual DCG delta
    # is very important to account for
    #
    # We can account for this when we normalize alpha and beta
    diff['grade_changed'] = False
    diff['actual_dcg_delta'] = actual_dcg_delta
    moves_up = diff['weight_delta'] > 0
    num_moves_up = len(diff.loc[moves_up])
    moves_down = diff['weight_delta'] < 0
    num_moves_down = len(diff.loc[moves_down])
    changed = diff['weight_delta'] != 0

    diff.loc[:, 'grade'] = 0
    diff.loc[changed, 'grade_changed'] = True

    start = perf_counter()

    learning_rate = 0.001
    rounds = 0
    while True:
        # Assign the items with a positive weight delta (moved UP) a relevance of 1
        # with probability `prob_positive` (and conversely for negatives)
        rand_grades_positive = _biased_random_sample(num_moves_up,
                                                     prob_of_relevance=prob_positive)
        rand_grades_negative = _biased_random_sample(num_moves_down,
                                                     prob_of_relevance=1.0 - prob_positive)

        diff.loc[moves_up, 'grade'] = rand_grades_positive
        diff.loc[moves_down, 'grade'] = rand_grades_negative

        # DCG delta of this simulated universe - how close is it to the observed DCG delta?
        simulated_dcg_delta = (diff['grade'] * diff['weight_delta']).sum()
        universe_prob = _universe_probability(actual_dcg_delta, simulated_dcg_delta,
                                              std_dev=dcg_diff_std_dev)

        # Increment alpha and beta in proportion to probability of the universe being real
        # how close to observed universe relative to how many possible universes could be THE universe (num_grades_changed)
        diff.loc[(diff['grade'] == 1) & changed, 'alpha'] += universe_prob
        diff.loc[(diff['grade'] == 0) & changed, 'beta'] += universe_prob

        diff.loc[changed, 'std_dev'] = np.sqrt((diff['alpha'] * diff['beta']) /
                                               (((diff['alpha'] + diff['beta'])**2) * (1 + diff['alpha'] + diff['beta'])))

        biggest_std_dev = diff.loc[changed, 'std_dev'].max()

        # increment in the direction of the weight delta
        # inversely proportion to the probability of the universe being real
        # (ie we keep the probability of positive prior close to real universes)
        delta = actual_dcg_delta - simulated_dcg_delta
        update_scaled = 1 - universe_prob
        update = learning_rate * update_scaled * _sign(delta)
        prob_positive += update
        plausible_universe_prob += universe_prob

        if universe_prob > best_universe_prob:
            best_universe_prob = universe_prob

        if verbose and rounds % 100 == 0:
            msg = f"Sim: {simulated_dcg_delta:.2f}, Act: {actual_dcg_delta:.2f}({dcg_diff_std_dev:.3f}),"
            msg += f"Prob: {universe_prob:.3f}, Tot: {plausible_universe_prob:.3f} | StdDev? {biggest_std_dev:.3f}"
            msg += f"| Upd {update:.3f}, Draw {prob_positive:.3f}"
            msg += f" | {num_grades_changed} changed | {rounds}/{perf_counter() - start:.3f}s"
            print(msg)
        rounds += 1

        if biggest_std_dev <= converge_std_dev and rounds >= min_rounds:
            break

    # alpha - proportion of plausible universes that have this grade as 1
    # beta - proportion of plausible universes that have this grade as 0
    # total_universe_prob - accumulated probability density of all explored universes
    diff['alpha'] /= plausible_universe_prob
    diff['beta'] /= plausible_universe_prob
    diff['total_universe_prob'] = plausible_universe_prob

    print(f"BestCase: {best_case_dcg_delta}; Observed: {actual_dcg_delta}")
    print(f"Likelihood: {likelihood_not_random:.2f} | Total Universe Prob: {plausible_universe_prob:.2f}")
    cols = ['QueryId', 'DocumentId', 'position_before', 'position_after', 'weight_delta', 'alpha', 'beta']
    print(diff[diff['grade_changed']][cols].sort_values('alpha', ascending=False))

    # _debug_query(diff, 0)
    # _debug_query(diff, 1)
    return diff
