import pandas as pd
import numpy as np
from scipy.stats import norm

from util import *
from rbt import RedBlackTree # Order statistics red-black tree


def baseline_unconstrained(left_region, right_region, quantifier, statement):
    """
    Baseline algorithm for calculating the unconstrained support.

    Args:
    -----
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `statement: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.

    Returns:
    --------
    `support: float`
        The support of the statement in the data.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    baseline_unconstrained(left, right, lambda x: x.A, (0, 2))
    >>> 0.333...
    """
    lower, upper = statement
    satisfied = 0
    for x1 in left_region.itertuples(index=False):
        for x2 in right_region.itertuples(index=False):
            if lower < quantifier(x1) - quantifier(x2) < upper:
                satisfied += 1
    return satisfied / (len(left_region) * len(right_region))


def exact_unconstrained(left_region, right_region, quantifier, statement):
    """
    Efficient algorithm for calculating the unconstrained support.

    Args:
    -----
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `statement: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.

    Returns:
    --------
    `support: float`
        The support of the statement in the data.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    exact_unconstrained(left, right, lambda x: x.A, (0, 2))
    >>> 0.333...
    """
    cumulative = np.array([quantifier(x1) for x1 in left_region.itertuples(index=False)])
    cumulative.sort()
    lower, upper = statement
    satisfied = 0
    for x2 in right_region.itertuples(index=False):
        fx2 = quantifier(x2)
        low_idx = cumulative.searchsorted(fx2 + lower, side='right')
        high_idx = cumulative.searchsorted(fx2 + upper, side='left')
        if low_idx < high_idx:
            satisfied += (high_idx - low_idx)
    return satisfied / (len(left_region) * len(right_region))





def baseline_constrained(left_region, right_region, quantifier, statement, constraints):
    """
    Baseline algorithm for calculating the constrained support.

    Args:
    -----
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `statement: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.
    `constraints: callable`
        A function that takes two records and returns a boolean value (indicating if the pair is valid).

    Returns:
    --------
    `support: float`
        The support of the statement in the data.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    constraints = lambda x1, x2: x1.B == x2.B
    baseline_constrained(left, right, lambda x: x.A, (-1, 1), constraints)
    >>> 0.5
    """
    lower, upper = statement
    satisfied, total = 0, 0
    for x1 in left_region.itertuples(index=False):
        valid_right_region = validate_region(x1, right_region, constraints)
        for x2 in valid_right_region.itertuples(index=False):
            total += 1
            if lower < quantifier(x1) - quantifier(x2) < upper:
                satisfied += 1
    return satisfied / total


def exact_constrained(left_region, right_region, quantifier, statement, constraints):
    """
    Efficient algorithm for calculating the constrained support.

    Args:
    -----
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `statement: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.
    `constraints: callable`
        A function that takes two records and returns a boolean value (indicating if the pair is valid).

    Returns:
    --------
    `support: float`
        The support of the statement in the data.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    constraints = lambda x1, x2: x1.B == x2.B
    exact_constrained(left, right, lambda x: x.A, (-1, 1), constraints)
    >>> 0.5
    """
    lower, upper = statement
    total, satisfied = 0, 0
    rbt = RedBlackTree()
    previous_valid_left_region = pd.DataFrame()
    for x2 in right_region.itertuples(index=False):
        current_valid_left_region = validate_region(x2, left_region, constraints)
        points_to_remove = previous_valid_left_region[~previous_valid_left_region.isin(current_valid_left_region).all(1)]
        points_to_add = current_valid_left_region[~current_valid_left_region.isin(previous_valid_left_region).all(1)]
        for x1 in points_to_remove.itertuples(index=False):
            rbt.delete(quantifier(x1))
        for x1 in points_to_add.itertuples(index=False):
            rbt.insert(quantifier(x1))
        previous_valid_left_region = current_valid_left_region
        total += rbt.size()
        fx2 = quantifier(x2)  
        low_idx = rbt.count_smaller_than(fx2 + lower)
        high_idx = rbt.count_smaller_than(fx2 + upper)
        if low_idx < high_idx:
            satisfied += (high_idx - low_idx)
    return satisfied / total


def pair_sampling(dataset, left_region, right_region, quantifier, statement, constraints, confidence):
    """
    Monte Carlo algorithm for calculating the constrained support by sampling pairs of records.

    Args:
    -----
    `dataset: pd.DataFrame`
        The data to sample from.
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `statement: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.
    `constraints: callable`
        A function that takes two records and returns a boolean value (indicating if the pair is valid).
    `confidence: float`
        The confidence level for the error.

    Returns:
    --------
    `support: float`
        The support of the statement in the data.
    `error: float`
        The error of the support estimate.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    constraints = lambda x1, x2: x1.B == x2.B
    pair_sampling(data, left, right, lambda x: x.A, (-1, 1), constraints, 0.95)
    >>> (0.5, 0.015...)
    """
    lower, upper = statement
    satisfied = 0
    for _ in range(len(dataset)):
        x2 = right_region.sample(n=1).iloc[0]
        valid_left_region = validate_region(x2, left_region, constraints)
        x1 = valid_left_region.sample(n=1).iloc[0]
        if lower < quantifier(x1) - quantifier(x2) < upper:
            satisfied += 1
    support = satisfied / len(dataset)    
    error = norm.ppf(1-confidence/2) * np.sqrt(support * (1 - support) / len(dataset))
    return support, error


def point_sampling(dataset, left_region, right_region, quantifier, statement):
    """
    Monte Carlo algorithm for calculating the unconstrained support by sampling subsets of the regions.

    Args:
    -----
    `dataset: pd.DataFrame`
        The data to sample from.
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `statement: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.

    Returns:
    --------
    `support: float`
        The support of the statement in the data.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    point_sampling(data, left, right, lambda x: x.A, (0, 2))
    >>> 0.375
    """
    lower, upper = statement
    satisfied = 0
    left_samples = left_region.sample(n=len(dataset), replace=True).itertuples(index=False)
    right_samples = right_region.sample(n=len(dataset), replace=True).itertuples(index=False)
    cumulative = np.array([quantifier(x1) for x1 in left_samples])
    cumulative.sort()
    for x2 in right_samples:
        fx2 = quantifier(x2)
        low_idx = cumulative.searchsorted(fx2 + lower, side='right')
        high_idx = cumulative.searchsorted(fx2 + upper, side='left')
        if low_idx < high_idx:
            satisfied += (high_idx - low_idx)
    return satisfied / len(dataset)**2


def tightest_statement(left_region, right_region, quantifier, support, constraints=None):
    """
    Calculates the smallest range of values that achieves the specified support.

    Args:
    -----
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `support: float` in (0, 1) exclusive
        The desired support level.
    `constraints: callable` or `None`
        A function that takes two records and returns a boolean value (indicating if the pair is valid).
        If `None`, the unconstrained support is calculated.

    Returns:
    --------
    `ts: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    left = rectangular_region(data, {'A': (2, 4)})
    right = rectangular_region(data, {'B': (5, 7)})
    tightest_statement(left, right, lambda x: x.A, 0.9, lambda x1, x2: x1.B == x2.B)
    >>> (-1, 1)
    """
    if constraints is None:
        differences = np.concatenate([[quantifier(x2) - quantifier(x1)
                                       for x1 in left_region.itertuples(index=False)]
                                       for x2 in right_region.itertuples(index=False)])
    else:
        differences = np.concatenate([[quantifier(x2) - quantifier(x1) 
                                       for x1 in validate_region(x2, left_region, constraints).itertuples(index=False)]
                                       for x2 in right_region.itertuples(index=False)])
    print(differences)
    differences.sort()
    n_trendlines = len(differences)
    window_size = int(n_trendlines * support)
    min_range_width = np.inf
    best_lower, best_upper = None, None
    for i in range(n_trendlines - window_size):
        lower, upper = differences[i], differences[i + window_size]
        range_width = upper - lower
        if range_width < min_range_width:
            min_range_width = range_width
            best_lower, best_upper = lower, upper
    return best_lower, best_upper


def most_supported_statement(left_region, right_region, quantifier, range_width, constraints=None):
    """
    Calculates the range of values that achieves the highest support.

    Args:
    -----
    `left_region: pd.DataFrame`
        The left (minued) region of the data.
    `right_region: pd.DataFrame`
        The right (subtrahend) region of the data.
    `quantifier: callable`
        A function that takes a record and returns a value.
    `range_width: float`
        The desired width of the range.
    `constraints: callable` or `None`
        A function that takes two records and returns a boolean value (indicating if the pair is valid).
        If `None`, the unconstrained support is calculated.

    Returns:
    --------
    `mss: tuple`
        A tuple of (lower, upper) bounds for the difference between the quantified values of the two regions.
    `support: float`
        The support this statement achieves over the data.

    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    left = rectangular_region(data, {'A': (1, 2)})
    right = rectangular_region(data, {'B': (5, 7)})
    most_supported_statement(left, right, lambda x: x.A, 2, lambda x1, x2: x1.B == x2.B)
    >>> ((-1.0, 1.0), 0.75)
    """
    if constraints is None:
        differences = np.concatenate([[quantifier(x2) - quantifier(x1)
                                       for x1 in left_region.itertuples(index=False)]
                                       for x2 in right_region.itertuples(index=False)])
    else:
        differences = np.concatenate([[quantifier(x2) - quantifier(x1)
                                       for x1 in validate_region(x2, left_region, constraints).itertuples(index=False)]
                                       for x2 in right_region.itertuples(index=False)])
    differences.sort()
    n_trendlines = len(differences)
    max_support = 0
    best_lower, best_upper = None, None
    for low_idx in range(n_trendlines):
        lower = differences[low_idx]
        high_idx = np.searchsorted(differences, lower + range_width, side='right') - 1
        support = high_idx - low_idx
        if support > max_support:
            max_support = support
            best_lower, best_upper = lower, differences[high_idx]
    return (best_lower, best_upper), (max_support / len(differences))
