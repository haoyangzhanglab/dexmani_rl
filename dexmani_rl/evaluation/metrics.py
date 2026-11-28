import numpy as np
from scipy.stats import binomtest


def compare(baseline, steered):
    if [x['seed'] for x in baseline] != [x['seed'] for x in steered]:
        raise ValueError('Paired evaluation requires the same seeds in the same order')
    count = len(baseline)
    if not count:
        return {'episodes': 0}
    a = np.array([x['success'] for x in baseline], dtype=int)
    b = np.array([x['success'] for x in steered], dtype=int)
    diff = b - a
    # Delta = P(steered-only success) - P(BC-only success). Bonferroni combines
    # two 97.5% Clopper-Pearson intervals into a conservative 95% interval.
    wins = binomtest(int((diff == 1).sum()), count).proportion_ci(.975, method='exact')
    losses = binomtest(int((diff == -1).sum()), count).proportion_ci(.975, method='exact')
    ci = [wins.low - losses.high, wins.high - losses.low]
    return {'episodes': count, 'baseline_successes': int(a.sum()),
            'steered_successes': int(b.sum()), 'success_rate_delta': float(diff.mean()),
            'paired_conservative_95_ci': ci, 'training_seeds': 1}


def evaluation_report(results, seeds):
    steered = results.get('steered', [])
    complete = [row['seed'] for row in results['baseline']] == list(seeds)
    result = dict(results, steered=steered, test_seeds=list(seeds), evaluation_complete=complete)
    if not complete:
        result['comparison'] = {'episodes': len(steered), 'requested_episodes': len(seeds),
                                'evaluation_complete': False}
        return result
    result['comparison'] = compare(results['baseline'], steered) if 'steered' in results else {'episodes': 0}
    return result
