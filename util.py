import pandas as pd 


def rectangular_region(data, bounds):
    """
    Returns a subset of the data where each column is within the specified bounds.
    
    Args:
    -----
    `data: pd.DataFrame`
        The data to filter.
    `bounds: dict`
        A mapping from column names to a tuple of (low, high) values.
        
    Returns:
    --------
    `rect_data: pd.DataFrame`
        A subset of the data's records where each column is within the specified bounds.
    
    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    rectangular_region(data, {'A': (1, 2), 'B': (5, 10)})
    >>>    A  B  C
        1  2  5  8
    ```
    """
    return data.copy().query(
        ' and '.join(f'{col} >= {low} and {col} <= {high}' 
                        for col, (low, high) in bounds.items())
    )


# def circular_region(data, centers, radius):
#     return data.copy().query(
#         ' and '.join(f'({col} - {center_val}) ** 2 <= {radius ** 2}' 
#                         for col, center_val in centers.items())
#     )


def circular_region(data, centers_radii):
    """
    Returns a subset of the data where each column is within the specified bounds.
    
    Args:
    -----
    `data: pd.DataFrame`
        The data to filter.
    `centers_radii: dict`
        A mapping from column names to a tuple of (center, radius) values.
        
    Returns:
    --------
    `circ_data: pd.DataFrame`
        A subset of the data's records where each column is within the specified bounds.
    
    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    circular_region(data, {'A': (1, 1), 'B': (6, 2)})
    >>>    A  B  C
        0  1  4  7
        1  2  5  8
    ```
    """
    return data.copy().query(
        ' and '.join(f'({col} - {center}) ** 2 <= {radius ** 2}' 
                        for col, (center, radius) in centers_radii.items())
    )


def validate_region(x1, region, constaints):
    """
    Returns a subset of the region that satisfies the constraints with respect to x1.

    Args:
    -----
    `x1: pd.Series` or `pd.Pandas`
        The record to validate against.
    `region: pd.DataFrame`
        The region to filter.
    `constraints: callable`
        A function that takes two records and returns a boolean value.

    Returns:
    --------
    `valid_region: pd.DataFrame`
        A subset of the region that satisfies the constraints with respect to x1.
    
    Example:
    --------
    ```python
    data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    x1 = pd.Series({'A': 11, 'B': 12, 'C': 13})
    validate_region(x1, data, lambda x1, x2: x1.A % 2 == x2.A % 2)
    >>>    A  B  C
        0  1  4  7
        2  3  6  9
    ```
    """
    valid_region = region.copy()
    for idx, x2 in region.iterrows():
        if not constaints(x1, x2):
            valid_region.drop(idx, inplace=True)
    return valid_region
