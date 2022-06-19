# Can we infer relevance grades from just a difference in the mean NDCG of two samples?
import pandas as pd
import numpy as np
from diff_simulation import estimate_relevance, create_results_diff, ideal_dcg_at_5, corpus


# Submissions from the kaggle vmware competition
# NDCG at 5
ndcgs = {
    'data/noise.csv': 0.00060,
    'data/use_feedback_rrf_turnbull_submission_1653226391886872.csv': 0.16806,
    'data/turnbull_submission_1652544680901428.csv': 0.20911,
    'data/pull_out_firstline_turnbull_1653253060074837.csv': 0.29668,
    'data/rerank_slop_search_remaining_lines_max_snippet_at_5_turnbull_1654439885030507.csv': 0.31574,
    'data/with_best_compounds_at_5_only_phrase_search_turnbull_1654439995765457.csv': 0.31643,
    'data/with_best_compounds_at_50_plus_10_times_use_turnbull_165445182567455.csv': 0.32681,

    # Random noise we assume has NDCG=0
    # 'noise.csv': 0.0
}


def main():
    judgments = pd.DataFrame(columns=['QueryId', 'DocumentId', 'alpha', 'beta',
                                      'weight_delta', 'position_delta'])
    # TODO - do more simulations for larger diff
    # Scale alpha and beta when more information is contained (ie closer to max diff)
    num_simulations = 100
    runs = 0
    # Cycle only through additive changes, not every combination, to ensure
    # we don't double-count change
    for results_before_csv, results_after_csv in zip(ndcgs.keys(), list(ndcgs.keys())[1:]):
        if results_before_csv == results_after_csv:
            continue
        mean_ndcg_diff = ndcgs[results_after_csv] - ndcgs[results_before_csv]
        print(results_before_csv, results_after_csv, num_simulations, f"diff: {mean_ndcg_diff:.3f}")

        results_before = pd.read_csv(results_before_csv)
        results_after = pd.read_csv(results_after_csv)

        results_diff = create_results_diff(results_before, results_after)

        # Translate our NDCG@5 to a sum of all DCG@5 to simplify the simulation
        actual_dcg_delta = len(results_diff['QueryId'].unique()) * mean_ndcg_diff * ideal_dcg_at_5

        results = estimate_relevance(results_diff,
                                     actual_dcg_delta=actual_dcg_delta,
                                     verbose=True)

        # Accumulate judgments from this pair into the evaluation
        assert (results[results['position_delta'] == 0]['weight_delta'] == 0).all()
        prob_not_random = results['prob_not_random'].iloc[0]
        results = results.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta']].sum()
        # Score on how relevant we think this simulation is to the overall alpha / beta
        results['alpha'] *= prob_not_random
        results['beta'] *= prob_not_random
        if len(judgments) == 0:
            judgments = results
        else:
            judgments = judgments.append(results)

        judgments = \
            judgments.groupby(['QueryId', 'DocumentId'])[['alpha', 'beta']].sum()
        print(len(results), '->', len(judgments))

        runs += 1

    # Runs likely repeat information between them. How do we
    # account for their indpendence (they are not entirely indepnedent)
    # judgments['alpha'] /= math.log(runs)
    # judgments['beta'] /= math.log(runs)

    # Compute a grade using alpha and beta
    judgments['grade'] = judgments['alpha'] / (judgments['alpha'] + judgments['beta'])
    judgments['grade_std_dev'] = np.sqrt((judgments['alpha'] * judgments['beta']) /
                                         (((judgments['alpha'] + judgments['beta'])**2) * (1 + judgments['alpha'] + judgments['beta'])))

    # Join with corpus for debugging
    queries = pd.read_csv('data/test.csv')
    judgments = judgments.reset_index()
    results = judgments.merge(corpus, right_on='f_name', left_on='DocumentId', how='left')
    results = results.merge(queries, on='QueryId', how='left')
    results.to_csv('simulated_results.csv', index=False)


if __name__ == "__main__":
    main()
