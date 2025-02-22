from typing import Dict, Tuple, Callable, Union, Any

import pandas as pd


def rectangular_region(data: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Filters the dataset to return records where each specified column value is within the given bounds.

    @param data: The dataset to filter.
    @param bounds: A dictionary mapping column names to (low, high) value ranges.
    @return: A subset of the dataset where each column value falls within the specified bounds.

    @example:
    >>> data_ = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> rectangular_region(data_, {'A': (1, 2), 'B': (5, 10)})
    A  B  C
    1  2  5  8
    """
    return data.copy().query(
        ' and '.join(f'{col} >= {low} and {col} <= {high}'
                     for col, (low, high) in bounds.items())
    )


def circular_region(data: pd.DataFrame, centers_radii: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Filters the dataset to return records where each specified column value is within a circular boundary.

    @param data: The dataset to filter.
    @param centers_radii: A dictionary mapping column names to (center, radius) values.
    @return: A subset of the dataset where each column value falls within the specified circular boundary.

    @example:
    >>> data_ = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> circular_region(data_, {'A': (1, 1), 'B': (6, 2)})
    A  B  C
    0  1  4  7
    1  2  5  8
    """
    return data.copy().query(
        ' and '.join(f'({col} - {center}) ** 2 <= {radius ** 2}'
                     for col, (center, radius) in centers_radii.items())
    )


def validate_region(
        x1: Union[Tuple[Any, ...], pd.Series],
        region: pd.DataFrame,
        constraints: Callable[[pd.Series, pd.Series], bool]
) -> pd.DataFrame:
    """
    Filters a region to return records that satisfy the constraints with respect to a given record x1.

    @param x1: The reference record to validate against.
    @param region: The dataset to filter.
    @param constraints: A function that evaluates whether a pair of records satisfies the constraints.
    @return: A subset of the region that satisfies the constraints.

    @example:
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> x = pd.Series({'A': 11, 'B': 12, 'C': 13})
    >>> validate_region(x, data, lambda x1_, x2_: x1_.A % 2 == x2_.A % 2)
    A  B  C
    0  1  4  7
    2  3  6  9
    """
    valid_region = region.copy()
    for idx, x2 in region.iterrows():
        if not constraints(x1, x2):
            valid_region.drop(idx, inplace=True)
    return valid_region
