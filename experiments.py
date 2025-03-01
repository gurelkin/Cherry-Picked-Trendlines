import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import support
from region_extractor import rectangular_region


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

    s5 = support.pair_sampling(data, left, right, lambda x: x.A, statement, constraints, 0.95)
    s6 = support.point_sampling(data, left, right, lambda x: x.A, statement)

    ts = support.tightest_statement(left, right, lambda x: x.A, 0.34, constraints)
    mss = support.most_supported_statement(left, right, lambda x: x.A, 2, constraints)

    print(f'Baseline Unconstrained: {s1}')
    print(f'Exact Unconstrained: {s2}')
    print(f'Baseline Constrained: {s3}')
    print(f'Exact Constrained: {s4}')
    print(f'Pair Sampling Constrained: {s5}')
    print(f'Point Sampling Unconstrained: {s6}')
    print(f'Tightest Statement: {ts}')
    print(f'Most Supported Statement: {mss}')


def beer_sheva_temps():
    data = pd.read_csv('data/beer-sheva_temps.csv')
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y')

    f_x = lambda x: x.temp
    statement = (0, math.inf)
    winter = rectangular_region(data, {'datetime': ('01/01/2023', '03/31/2023')})
    summer = rectangular_region(data, {'datetime': ('06/01/2023', '08/31/2023')})
    constraints = lambda x1, x2: True

    s1 = support.baseline_unconstrained(summer, winter, f_x, statement)
    s2 = support.exact_unconstrained(summer, winter, f_x, statement)
    s3 = support.point_sampling(data, summer, winter, f_x, statement)
    ts = support.tightest_statement(summer, winter, f_x, 0.95, constraints)

    print('Statement: In Beer-Sheva the summer is hotter than the winter')
    print(f'Baseline Unconstrained: {s1 * 100:.2f}%')
    print(f'Exact Unconstrained: {s2 * 100:.2f}%')
    print(f'Point Sampling Unconstrained: {s3 * 100:.2f}%')
    print(f'Tightest Statement (at least 95% support): {ts}')


def beer_sheva_temps_ts_graph():
    data = pd.read_csv('data/beer-sheva_temps.csv')
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y')

    f_x = lambda x: x.temp
    winter = rectangular_region(data, {'datetime': ('01/01/2023', '03/31/2023')})
    summer = rectangular_region(data, {'datetime': ('06/01/2023', '08/31/2023')})
    constraints = lambda x1, x2: True

    ts_per_support = {
        s / 100: support.tightest_statement(summer, winter, f_x, s / 100, constraints)
        for s in range(0, 100, 5)
    }

    # Extract support values (x-axis)
    supports = list(ts_per_support.keys())

    # Extract interval ranges (y-axis)
    y_start = [interval[0] for interval in ts_per_support.values()]  # Start of the interval
    y_end = [interval[1] for interval in ts_per_support.values()]  # End of the interval

    # Create a figure and axis
    plt.figure(figsize=(8, 5))

    # Plot sticks (vertical lines from x_start to x_end at each support point)
    plt.vlines(supports, y_start, y_end, colors='b', linewidth=2, label="Tightest Statement")

    # Customize the graph
    plt.xlabel("Support")
    plt.ylabel("Tightest Statement")
    plt.title("Tightest Statement Per Support")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show the plot
    plt.show()


def beer_sheva_temps_mss_graph():
    data = pd.read_csv('data/beer-sheva_temps.csv')
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y')

    f_x = lambda x: x.temp
    winter = rectangular_region(data, {'datetime': ('01/01/2023', '03/31/2023')})
    summer = rectangular_region(data, {'datetime': ('06/01/2023', '08/31/2023')})
    constraints = lambda x1, x2: True

    mss_per_range = {
        r: support.most_supported_statement(summer, winter, f_x, r, constraints)
        for r in range(0, 50, 1)
    }

    # Extract data
    supports = list(mss_per_range.keys())  # x positions (support ranges)
    statement_ranges = [val[0] for val in mss_per_range.values()]  # Extracted statements
    support_values = [val[1] for val in mss_per_range.values()]  # Support values

    # Convert range keys to single representative X positions (midpoint for visualization)
    x_positions = [np.mean(s) for s in supports]

    # Extract statement start and end points
    y_start = [s[0] for s in statement_ranges]
    y_end = [s[1] for s in statement_ranges]

    # Create a figure with 2 subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

    # First plot: Blue vertical sticks for Statements
    axes[0].vlines(x_positions, y_start, y_end, colors='b', linewidth=2, label="Statement Ranges")
    axes[0].set_ylabel("Statement Range")
    axes[0].set_title("Most Supported Statement Per Statement Range")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Second plot: Red bars for Support values
    axes[1].bar(x_positions, support_values, width=0.2, color='r', alpha=0.6, label="Support Values")
    axes[1].set_xlabel("Statement Ranges")
    axes[1].set_ylabel("Support Value")
    axes[1].set_title("Support Per Statement Range")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def dead_sea_level():
    data = pd.read_csv('data/dead_sea.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    f_x = lambda x: x.SeaLevel
    statement = (0, math.inf)
    before_1991 = rectangular_region(data, {'Date': ('01/01/1900', '12/31/1991')})
    after_1991 = rectangular_region(data, {'Date': ('01/01/1992', '12/31/2025')})
    constraints = lambda x1, x2: True

    s1 = support.baseline_unconstrained(after_1991, before_1991, f_x, statement)
    s2 = support.exact_unconstrained(after_1991, before_1991, f_x, statement)
    s3 = support.point_sampling(data, after_1991, before_1991, f_x, statement)
    mss = support.most_supported_statement(after_1991, before_1991, f_x, 30, constraints)

    print('Statement: Since 1991 the Dead Sea level is on the rise')
    print(f'Baseline Unconstrained: {s1 * 100:.2f}%')
    print(f'Exact Unconstrained: {s2 * 100:.2f}%')
    print(f'Point Sampling Unconstrained: {s3 * 100:.2f}%')
    print(f'Most Supported Statement (for 30 meters difference): Statement={mss[0]}, Support={mss[1] * 100:.2f}%')


def african_gdp():
    data = pd.read_csv('data/africa_gdp.csv')

    f_x = lambda x: x.GDP
    statement = (0, math.inf)
    gdp_2008 = rectangular_region(data, {'Year': (2008, 2008)})
    gdp_2009 = rectangular_region(data, {'Year': (2009, 2009)})
    constraints = lambda x1, x2: x1.Country == x2.Country

    s1 = support.baseline_constrained(gdp_2008, gdp_2009, f_x, statement, constraints)
    s2 = support.exact_constrained(gdp_2008, gdp_2009, f_x, statement, constraints)
    s3 = support.pair_sampling(data, gdp_2008, gdp_2009, f_x, statement, constraints, 0.95)
    ts = support.tightest_statement(gdp_2008, gdp_2009, f_x, 0.95, constraints)
    mss = support.most_supported_statement(gdp_2008, gdp_2009, f_x, 1_000_000_000, constraints)

    print('Statement: African countries did not suffer from the great recession of 2008, '
          'i.e. their GDP did not decreased from 2008 to 2009')
    print(f'Baseline Constrained: {s1 * 100:.2f}%')
    print(f'Exact Constrained: {s2 * 100:.2f}%')
    print(f'Pair Sampling Constrained: Support={s3[0] * 100:.2f}%, Error Margin={s3[1] * 100:.2f}%')
    print(f'Tightest Statement (at least 95% support): {ts}')
    print(f'Most Supported Statement(for 1B$ difference): Statement={mss[0]}, Support={mss[1] * 100:.2f}%')


def germany_rainfall():
    data = pd.read_csv('data/rainfall_germany.csv')

    f_x = lambda x: x.Rainfall
    statement = (-2.5, 2.5)
    dusseldorf = rectangular_region(data, {'City': ('Dusseldorf', 'Dusseldorf')})
    berlin = rectangular_region(data, {'City': ('Berlin', 'Berlin')})
    constraints = lambda x1, x2: x1.Month == x2.Month and x1.Year == x2.Year

    s1 = support.baseline_constrained(dusseldorf, berlin, f_x, statement, constraints)
    s2 = support.exact_constrained(dusseldorf, berlin, f_x, statement, constraints)
    s3 = support.pair_sampling(data, dusseldorf, berlin, f_x, statement, constraints, 0.95)
    mss = support.most_supported_statement(dusseldorf, berlin, f_x, 30, constraints)

    print(
        'Statement: Dusseldorf and Berlin has approximately the same rain (up to 2.5mm difference) amount for the same month.')
    print(f'Baseline Constrained: {s1 * 100:.2f}%')
    print(f'Exact Constrained: {s2 * 100:.2f}%')
    print(f'Pair Sampling Constrained: Support={s3[0] * 100:.2f}%, Error Margin={s3[1] * 100:.2f}%')
    print(f'Most Supported Statement (for 30mm difference): Statement={mss[0]}, Support={mss[1] * 100:.2f}%')


def danish_house():
    data = pd.read_csv('data/danish_house.csv')

    f_x = lambda x: x.purchase_price
    old = rectangular_region(data, {'year_build': (-math.inf, 1899)})
    new = rectangular_region(data, {'year_build': (1900, math.inf)})
    statement = (0, math.inf)

    s1 = support.point_sampling(data, old, new, f_x, statement)
    print(
        'Statement: In Denmark the prices of old (built before 1900) house are higher than new houses (built from 1900).'
    )
    print(f'Point Sampling Unconstrained: {s1 * 100:.2f}%')


def stochastic_danish_house():
    data = pd.read_csv('data/danish_house.csv')
    danish_house_subset = data.sample(n=10_000, random_state=42)

    f_x = lambda x: x.purchase_price
    old = rectangular_region(danish_house_subset, {'year_build': (-math.inf, 1899)})
    new = rectangular_region(danish_house_subset, {'year_build': (1900, math.inf)})
    statement = (0, math.inf)

    percentages = list(range(5, 101, 5))
    point_sampling_results = []
    exact_unconstrained_results = []

    exact_unconstrained_total = support.exact_unconstrained(old, new, f_x, statement)

    for p in percentages:
        sample_size = round(p / 100 * len(danish_house_subset))
        sample_data = danish_house_subset.sample(n=sample_size, random_state=42)

        old_sample = rectangular_region(sample_data, {'year_build': (-math.inf, 1899)})
        new_sample = rectangular_region(sample_data, {'year_build': (1900, math.inf)})

        s_point = support.point_sampling(sample_data, old_sample, new_sample, f_x, statement)
        s_exact_unconstrained = support.exact_unconstrained(old_sample, new_sample, f_x, statement)

        point_sampling_results.append(s_point)
        exact_unconstrained_results.append(s_exact_unconstrained)

    plt.figure(figsize=(8, 6))
    plt.plot(percentages, point_sampling_results, label='Point Sampling', color='blue', marker='o')
    plt.plot(percentages, exact_unconstrained_results, label='Exact Unconstrained', color='green', marker='s')
    plt.axhline(y=exact_unconstrained_total, color='red', linestyle='--', label='Exact Unconstrained (Full Dataset)')

    plt.xlabel("Sample Percentage Size")
    plt.ylabel("Support Value")
    plt.title("Exact VS Point Sampling")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
