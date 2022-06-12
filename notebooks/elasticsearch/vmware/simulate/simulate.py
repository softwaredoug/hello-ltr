# Can we infer relevance grades from just a difference in the mean NDCG of two samples?
import pandas as pd
import numpy as np
import os
import pickle
import random
from statistics import NormalDist


# Submissions from the kaggle vmware competition
# NDCG at 5
ndcgs = {
    'data/use_feedback_rrf_turnbull_submission_1653226391886872.csv': 0.16806,
    'data/turnbull_submission_1652544680901428.csv': 0.20911,
    'data/pull_out_firstline_turnbull_1653253060074837.csv': 0.29668,
    'data/rerank_slop_search_remaining_lines_max_snippet_at_5_turnbull_1654439885030507.csv': 0.31574,
    'data/with_best_compounds_at_5_only_phrase_search_turnbull_1654439995765457.csv': 0.31643,
    'data/with_best_compounds_at_50_plus_10_times_use_turnbull_165445182567455.csv': 0.32681,
    # 'data/simulated_labels_turnbull.csv': 0.26169,
    # 'data/simulated_labels_turnbull_2.csv': 0.28722,

    # Random noise we assume has NDCG=0
    # 'noise.csv': 0.0
}

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


def sign(x):
    return -1.0 if x < 0.0 else 1.0


def biased_random_sample(sample_size, prob_of_relevance=0.9):
    sample = np.random.random_sample(sample_size)
    biased_sample = sample.copy()
    biased_sample[sample < prob_of_relevance] = 1
    biased_sample[sample >= prob_of_relevance] = 0
    return biased_sample.astype(int)


def add_weights(results):
    """Add a position. Compute DCG weights"""
    results['position'] = results.groupby('QueryId').cumcount() + 1
    results['weight'] = 1 / np.log2(results['position'] + 1)
    return results


def create_results_diff(results_before, results_after):
    """Compute the DCG delta weights resulting from the before and after,
       so we can compare to observed mean delta dcg."""

    results_before = results_before.groupby('QueryId').head(5)
    results_after = results_after.groupby('QueryId').head(5)
    assert len(results_before) == len(results_after)

    results_before = add_weights(results_before)
    results_after = add_weights(results_after)
    results = results_before.merge(results_after,
                                   on=['QueryId', 'DocumentId'],
                                   how='outer')
    results = results.rename(
        columns={'position_x': 'position_before',
                 'position_y': 'position_after',
                 'weight_x': 'weight_before',
                 'weight_y': 'weight_after'}
    )
    # For each document, its DCG weight before and aftter
    results['weight_after'] = results['weight_after'].replace(np.nan, 0)
    results['weight_before'] = results['weight_before'].replace(np.nan, 0)
    results['weight_delta'] = results['weight_after'] - results['weight_before']
    results['position_delta'] = results['position_after'] - results['position_before']
    results['weight_delta_abs'] = np.abs(results['weight_delta'])

    return results


def universe_probability(actual_dcg_delta, simulated_dcg_delta, std_dev=1000):
    actual_universe_distribution = NormalDist(mu=actual_dcg_delta, sigma=std_dev)
    simulated_universe_distribution = NormalDist(mu=simulated_dcg_delta, sigma=std_dev)
    universe_prob = actual_universe_distribution.overlap(simulated_universe_distribution)

    return universe_prob


# To get the best universe, it must match the intersection of every diffs best universe
#
# So in reality, a good universe is one that satisfies the following:
#
#  prob(universe1) * prob(universe2) * ... * prob(universeN)
#
#
# Alternatively:
#
#   Find best universes for each diff, with probability
#   Intersect with next universe
#
#
#
#
def simulate_at(results, prob_positive, actual_dcg_delta, rounds=10, verbose=False,
                dcg_diff_std_dev=20, universe_prob_threshold=0.5):
    """Simulate by giving each doc moved up a grade of 1 with provided probabiility."""
    best_universe_prob = 0.0
    best_prob_positive = prob_positive
    universes = []
    for i in range(0, rounds):
        # Assign the items with a positive weight delta (moved UP) a relevance of 1
        # with probability `prob_positive` (and conversely for negatives)
        rand_grades_positive = biased_random_sample(len(results[results['weight_delta'] > 0]),
                                                    prob_of_relevance=prob_positive)
        rand_grades_negative = biased_random_sample(len(results[results['weight_delta'] < 0]),
                                                    prob_of_relevance=1.0 - prob_positive)

        results['grade'] = 0
        results['grade_changed'] = False
        results['actual_dcg_delta'] = actual_dcg_delta
        results.loc[results['weight_delta'] > 0, 'grade'] = rand_grades_positive
        results.loc[results['weight_delta'] < 0, 'grade'] = rand_grades_negative
        results.loc[results['weight_delta'] > 0, 'grade_changed'] = True
        results.loc[results['weight_delta'] < 0, 'grade_changed'] = True

        # DCG delta of this simulated universe - how close is it to the observed DCG delta?
        simulated_dcg_delta = sum(results['grade'] * results['weight_delta'])
        universe_prob = universe_probability(actual_dcg_delta, simulated_dcg_delta,
                                             std_dev=dcg_diff_std_dev)

        # Move probability of drawing positive (the prob weights > 0 have relevance=1) in
        # a direction closer to making the simulated relevance universe more likely
        tight_universe_prob = universe_probability(actual_dcg_delta, simulated_dcg_delta, std_dev=2)
        delta = actual_dcg_delta - simulated_dcg_delta
        update = 0.01 * (1 - (tight_universe_prob**0.000000001)) * sign(delta)
        prob_positive += update

        if universe_prob > best_universe_prob:
            best_universe_prob = universe_prob
            best_prob_positive = prob_positive

        if universe_prob > universe_prob_threshold:
            universe = results.copy()
            universe = universe.groupby(['QueryId', 'DocumentId']).first()
            universe['score'] = universe_prob
            universes.append(universe)

        if verbose:
            msg = f"Sim: {simulated_dcg_delta:.2f}, Act: {actual_dcg_delta:.2f}, Prob: {universe_prob:.3f} "
            msg += f"| Upd {update}, Draw {prob_positive:.3f} | Best {best_prob_positive:.3f} {best_universe_prob:.3f}"
            print(msg)

    universes = sorted(universes, key=lambda x: x['score'].iloc[0], reverse=True)

    return universes


def is_duplicate_universe(universe1, universe2):
    """Remove duplicate universes"""
    return universe1.groupby(['QueryId', 'DocumentId'])['grade'] == universe2.groupby(['QueryId', 'DocumentId'])['grade']


def intersect_universes(new_universes, universes, dcg_diff_std_dev=20, max_overlap=5000):
    """The best set of relevance grades corresponds to those where more diffs have accurate relevance grades
       so we want to get the intersection of all of the universes where we have good grades."""
    if universes == []:
        return new_universes

    intersected_universes = []

    print(f"Intersecting {len(universes)} universes")

    # What's the probability of new universes grades for existing universes delta?
    for old_universe in universes:
        for new_universe in new_universes:
            actual_dcg_delta = old_universe['actual_dcg_delta'].iloc[0]
            assert (old_universe['actual_dcg_delta'] == actual_dcg_delta).all()

            old_with_new_grades = old_universe.merge(new_universe,
                                                     on=['QueryId', 'DocumentId'],
                                                     how='left',
                                                     suffixes=('', '_new'))

            # New universes has nothing to say on this, so we'll just use the old universe grades
            assert (old_with_new_grades['grade_changed_new'].isna()
                    == old_with_new_grades['grade_new'].isna()).all()
            old_with_new_grades['grade_changed_new'] = old_with_new_grades['grade_changed_new'].fillna(False)
            changed_grades = old_with_new_grades['grade_changed_new']
            old_with_new_grades.loc[~changed_grades, 'grade_new'] = old_with_new_grades[~changed_grades]['grade']
            assert not (old_with_new_grades['grade_new'].isna()).all()

            # When the two universes have large and similar search spaces,
            # its difficult to find a good intersection, because the optimum for each is likely
            # to be heavily overfitted
            #
            # We dont update their relative probabilitiy
            overlapping_space = old_with_new_grades[
                (old_with_new_grades['weight_delta'] != 0) & old_with_new_grades['grade_changed_new']
            ][['weight_delta', 'grade_changed_new', 'grade', 'grade_new']]

            if len(overlapping_space) < max_overlap:
                simulated_dcg_delta = sum(old_with_new_grades['grade_new'] * old_with_new_grades['weight_delta'])

                new_universe_prob = universe_probability(actual_dcg_delta, simulated_dcg_delta, std_dev=dcg_diff_std_dev)
                print(f"Ovlap {len(overlapping_space)} Old {old_universe['score'].iloc[0]:.3f} Comb {new_universe_prob:.5f} | "
                      f"{simulated_dcg_delta:.3f} {actual_dcg_delta:.3f}")
                old_with_new_grades['score'] *= new_universe_prob
                # old_with_new_grades = old_with_new_grades.loc[old_with_new_grades['grade_changed'], :].copy()
            else:
                print(f"Not intersecting with {len(overlapping_space)} overlapping examples")

            # old_with_new_grades.loc[changed_grades, 'grade_changed'] = True
            old_with_new_grades = old_with_new_grades.loc[:, ~old_with_new_grades.columns.str.contains("_new$")]
            intersected_universes.append(old_with_new_grades)

    intersected_universes = sorted(intersected_universes, key=lambda x: x['score'].iloc[0], reverse=True)

    # Scale all universes to probability score of 1
    # max_universe_prob = intersected_universes[0]['score'].iloc[0]
    # if max_universe_prob > 0:
    #    for universe in intersected_universes:
    #        universe['score'] /= max_universe_prob

    # Dedup universes
    # deduped_universes = []
    # for universe1 in intersected_universes:
    #    for universe2 in intersected_universes:
    #        if is_duplicate_universe(universe1, universe2):
    #            print("Skiiping duplicate universe")
    #            continue
    #        deduped_universes.append(universe1)
    # intersected_universes = deduped_universes

    # How many universes are unique?
    print("Num universes:", len(intersected_universes))
    return intersected_universes


def main():
    judgments = pd.DataFrame(columns=['QueryId', 'DocumentId', 'alpha', 'beta',
                                      'weight_delta', 'position_delta'])
    num_simulations = 1000
    universes = []
    for results_before_csv, results_after_csv in zip(ndcgs.keys(), list(ndcgs.keys())[1:]):
        if results_before_csv == results_after_csv:
            continue
        mean_ndcg_diff = ndcgs[results_after_csv] - ndcgs[results_before_csv]
        print(results_before_csv, results_after_csv, num_simulations, f"diff: {mean_ndcg_diff:.3f}")

        results_before = pd.read_csv(results_before_csv)
        results_after = pd.read_csv(results_after_csv)

        results_diff = create_results_diff(results_before, results_after)

        # Translate our NDCG@5 to a DCG@5 to simplify the simulation
        actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5

        # Very weak prior, mean of 0.3
        if 'alpha' not in results_diff.columns:
            results_diff.loc[:, 'alpha'] = 0.01
            results_diff.loc[:, 'beta'] = 0.02

        prob_positive = random.uniform(0.24, 0.80)
        new_universes = simulate_at(results_diff, prob_positive,
                                    rounds=num_simulations,
                                    actual_dcg_delta=actual_dcg_delta,
                                    verbose=True)
        pickle.dump(new_universes, open(f"{os.path.basename(results_after_csv)}_to_{os.path.basename(results_before_csv)}_universes.pkl", "wb"))

        universes = intersect_universes(new_universes, universes)

        # Accumulate judgments from this pair into the evaluation
        # results = results.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta', 'weight_delta', 'position_delta']].sum()
        # judgments = pd.concat([judgments, results])
        # judgments = \
        #     judgments.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta', 'weight_delta', 'position_delta']].sum()
        # print(len(results), '->', len(judgments))

        # runs += 1

    # Runs likely repeat information between them. How do we
    # account for their indpendence (they are not entirely indepnedent)
    # judgments['alpha'] /= math.log(runs)
    # judgments['beta'] /= math.log(runs)
    import pdb; pdb.set_trace()
    judgments = universes[0]

    # Compute a grade using alpha and beta
    judgments['grade'] = judgments['alpha'] / (judgments['alpha'] + judgments['beta'])

    # Join with corpus for debugging
    corpus = pd.read_csv('data/vmware_ir_content.csv.zip', compression='zip')
    queries = pd.read_csv('data/test.csv')
    judgments = judgments.reset_index()
    results = judgments.merge(corpus, right_on='f_name', left_on='DocumentId', how='left')
    results = results.merge(queries, on='QueryId', how='left')
    results.to_csv('simulated_results.csv', index=False)


if __name__ == "__main__":
    main()
