from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from rbt import RedBlackTree  # Order statistics red-black tree
from utils import validate_region, rectangular_region


def baseline_unconstrained(
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        statement: Tuple[float, float]
) -> float:
    """
    Computes the unconstrained support using a brute-force baseline approach.

    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param statement: A tuple representing the lower and upper bounds for comparison.
    @return: The computed support as a float.

    Example:
    ```python
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> baseline_unconstrained(left, right, lambda x: x.A, (0, 2))
    0.333...
    """
    lower, upper = statement
    satisfied = 0
    for x1 in left_region.itertuples(index=False):
        for x2 in right_region.itertuples(index=False):
            if lower < quantifier(x1) - quantifier(x2) < upper:
                satisfied += 1
    return satisfied / (len(left_region) * len(right_region))


def exact_unconstrained(
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        statement: Tuple[float, float]
) -> float:
    """
    Computes the unconstrained support efficiently using sorted cumulative values.

    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param statement: A tuple representing the lower and upper bounds for comparison.
    @return: The computed support as a float.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> exact_unconstrained(left, right, lambda x: x.A, (0, 2))
    0.333...
    """
    lower, upper = statement
    cumulative = np.array([quantifier(x1) for x1 in left_region.itertuples(index=False)])
    cumulative.sort()
    satisfied = 0
    for x2 in right_region.itertuples(index=False):
        fx2 = quantifier(x2)
        low_idx = cumulative.searchsorted(fx2 + lower, side='right')
        high_idx = cumulative.searchsorted(fx2 + upper, side='left')
        if low_idx < high_idx:
            satisfied += (high_idx - low_idx)
    return satisfied / (len(left_region) * len(right_region))


def baseline_constrained(
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        statement: Tuple[float, float],
        constraints: Callable[[pd.Series, pd.Series], bool]
) -> float:
    """
    Computes the constrained support using a brute-force baseline approach.

    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param statement: A tuple representing the lower and upper bounds for comparison.
    @param constraints: A function that returns True if the pair satisfies the given constraints.
    @return: The computed support as a float.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> constraints = lambda x1, x2: x1.B == x2.B
    >>> baseline_constrained(left, right, lambda x: x.A, (-1, 1), constraints)
    0.5
    """
    lower, upper = statement
    satisfied, total = 0, 0
    for x1 in left_region.itertuples(index=False):
        valid_right_region = validate_region(x1, right_region, constraints)
        for x2 in valid_right_region.itertuples(index=False):
            total += 1
            if lower < quantifier(x1) - quantifier(x2) < upper:
                satisfied += 1
    return satisfied / total if total > 0 else 0


def exact_constrained(
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        statement: Tuple[float, float],
        constraints: Callable[[pd.Series, pd.Series], bool]
) -> float:
    """
    Computes the constrained support efficiently using an order statistics red-black tree.

    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param statement: A tuple representing the lower and upper bounds for comparison.
    @param constraints: A function that returns True if the pair satisfies the given constraints.
    @return: The computed support as a float.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> constraints = lambda x1, x2: x1.B == x2.B
    >>> exact_constrained(left, right, lambda x: x.A, (-1, 1), constraints)
    0.5
    """
    lower, upper = statement
    total, satisfied = 0, 0
    rbt = RedBlackTree()
    previous_valid_left_region = pd.DataFrame()
    for x2 in right_region.itertuples(index=False):
        current_valid_left_region = validate_region(x2, left_region, constraints)
        points_to_remove = previous_valid_left_region[
            ~previous_valid_left_region.isin(current_valid_left_region).all(1)]
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
    return satisfied / total if total > 0 else 0


def pair_sampling(
        dataset: pd.DataFrame,
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        statement: Tuple[float, float],
        constraints: Callable[[pd.Series, pd.Series], bool],
        confidence: float
) -> Tuple[float, float]:
    """
    Estimates the constrained support via Monte Carlo pair sampling.

    @param dataset: The full dataset used for sampling.
    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param statement: A tuple representing the lower and upper bounds for comparison.
    @param constraints: A function that returns True if the pair satisfies the given constraints.
    @param confidence: The confidence level for the error estimation.
    @return: A tuple containing the estimated support and its error margin.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> constraints = lambda x1, x2: x1.B == x2.B
    >>> pair_sampling(data, left, right, lambda x: x.A, (-1, 1), constraints, 0.95)
    (0.5, 0.015...)
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
    error = norm.ppf(1 - confidence / 2) * np.sqrt(support * (1 - support) / len(dataset))
    return support, error


def point_sampling(
        dataset: pd.DataFrame,
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        statement: Tuple[float, float]
) -> float:
    """
    Estimates the unconstrained support via Monte Carlo point sampling.

    @param dataset: The full dataset used for sampling.
    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param statement: A tuple representing the lower and upper bounds for comparison.
    @return: The estimated support as a float.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> point_sampling(data, left, right, lambda x: x.A, (0, 2))
    0.375
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
    return satisfied / (len(dataset) ** 2)


def tightest_statement(
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        support: float,
        constraints: Optional[Callable[[pd.Series, pd.Series], bool]] = None
) -> Tuple[float, float]:
    """
    Finds the smallest range of values that achieves the specified support.

    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param support: The desired support level (between 0 and 1 exclusive).
    @param constraints: Optional function to enforce additional constraints on valid pairs.
    @return: A tuple representing the lower and upper bounds for the optimal statement range.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    >>> left = rectangular_region(data, {'A': (2, 4)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> tightest_statement(left, right, lambda x: x.A, 0.9, lambda x1, x2: x1.B == x2.B)
    (-1, 1)
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


def most_supported_statement(
        left_region: pd.DataFrame,
        right_region: pd.DataFrame,
        quantifier: Callable[[pd.Series], float],
        range_width: float,
        constraints: Optional[Callable[[pd.Series, pd.Series], bool]] = None
) -> Tuple[Tuple[float, float], float]:
    """
    Finds the range of values that achieves the highest support.

    @param left_region: The left region (minuend) of the dataset.
    @param right_region: The right region (subtrahend) of the dataset.
    @param quantifier: A function that extracts a numerical value from a record.
    @param range_width: The fixed width of the range to evaluate.
    @param constraints: Optional function to enforce additional constraints on valid pairs.
    @return: A tuple where the first element is the optimal (lower, upper) range,
             and the second element is the support level for this range.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 5, 6, 6]})
    >>> left = rectangular_region(data, {'A': (1, 2)})
    >>> right = rectangular_region(data, {'B': (5, 7)})
    >>> most_supported_statement(left, right, lambda x: x.A, 2, lambda x1, x2: x1.B == x2.B)
    ((-1.0, 1.0), 0.75)
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
