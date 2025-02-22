from pandas import DataFrame

import support
from utils import rectangular_region


def example_usage():
    # A   B
    # 1   5
    # 2   6
    # 3   7
    # 4   8
    data = DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})

    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})

    s1 = support.baseline_unconstrained(left, right, lambda x: x.A, (0, 2))
    s2 = support.exact_unconstrained(left, right, lambda x: x.A, (0, 2))

    constraints = lambda x1, x2: True
    s3 = support.baseline_constrained(left, right, lambda x: x.A, (0, 2), constraints)
    s4 = support.exact_constrained(left, right, lambda x: x.A, (0, 2), constraints)

    s5 = support.pair_sampling(data, left, right, lambda x: x.A, (0, 2), constraints, 1)
    s6 = support.point_sampling(data, left, right, lambda x: x.A, (0, 2))

    ts = support.tightest_statement(left, right, lambda x: x.A, 0.34, constraints)
    mss = support.most_supported_statement(left, right, lambda x: x.A, 2, constraints)

    print(f'Baseline Unconstrained: {s1}')
    print(f'Exact Unconstrained: {s2}')
    print(f'Baseline Constrained: {s3}')
    print(f'Exact Constrained: {s4}')
    print(f'Pair Sampling: {s5}')
    print(f'Point Sampling: {s6}')
    print(f'Tightest Statement: {ts}')
    print(f'Most Supported Statement: {mss}')

if __name__ == '__main__':
    example_usage()
