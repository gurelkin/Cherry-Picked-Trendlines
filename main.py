import math

import pandas as pd
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
    statement = (0, 2)
    constraints = lambda x1, x2: x1.A % 2 != x2.A % 2
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})

    s1 = support.baseline_unconstrained(left, right, lambda x: x.A, statement)
    s2 = support.exact_unconstrained(left, right, lambda x: x.A, statement)

    s3 = support.baseline_constrained(left, right, lambda x: x.A, statement, constraints)
    s4 = support.exact_constrained(left, right, lambda x: x.A, statement, constraints)

    s5 = support.pair_sampling(data, left, right, lambda x: x.A, statement, constraints, 1)
    s6 = support.point_sampling(data, left, right, lambda x: x.A, statement)

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


def beer_sheva_temps():
    data = pd.read_csv('data/beer-sheva_temps.csv')
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y')

    f_x = lambda x: x.temp
    statement = (0, math.inf)
    constraints = lambda x1, x2: True
    winter = rectangular_region(data, {'datetime': ('01/01/2023', '03/31/2023')})
    summer = rectangular_region(data, {'datetime': ('06/01/2023', '08/31/2023')})

    s1 = support.baseline_unconstrained(summer, winter, f_x, statement)
    s2 = support.exact_unconstrained(summer, winter, f_x, statement)

    s3 = support.baseline_constrained(summer, winter, f_x, statement, constraints)
    s4 = support.exact_constrained(summer, winter, f_x, statement, constraints)

    s5 = support.pair_sampling(data, summer, winter, f_x, statement, constraints, 1)
    s6 = support.point_sampling(data, summer, winter, f_x, statement)

    ts = support.tightest_statement(summer, winter, f_x, 0.95, constraints)
    mss = support.most_supported_statement(summer, winter, f_x, 15, constraints)

    print('Statement: In Beer-Sheva the summer is hotter than the winter')
    print(f'Baseline Unconstrained: {s1 * 100:.2f}%')
    print(f'Exact Unconstrained: {s2 * 100:.2f}%')
    print(f'Baseline Constrained: {s3 * 100:.2f}%')
    print(f'Exact Constrained: {s4 * 100:.2f}%')
    print(f'Pair Sampling: Support={s5[0] * 100:.2f}%, Error Margin={s5[1] * 100:.2f}%')
    print(f'Point Sampling: {s6 * 100:.2f}%')
    print(f'Tightest Statement (for 15 degree difference): {ts}')
    print(f'Most Supported Statement: Statement={mss[0]}, Support={mss[1] * 100:.2f}%')


if __name__ == '__main__':
    # example_usage()
    beer_sheva_temps()
