"""
Data Normalization Techniques

Normalization scales data to a fixed range, typically [0, 1].
This is useful when features have different scales and you want to
bring them to a common scale without distorting differences in ranges.
"""

import numpy as np


def min_max_normalization(data: list[float] | np.ndarray) -> np.ndarray:
    """
    Normalize data using Min-Max normalization (scales to [0, 1] range).

    Formula: X_normalized = (X - X_min) / (X_max - X_min)

    Args:
        data: Input data as list or numpy array

    Returns:
        Normalized data in range [0, 1]

    Examples:
        >>> min_max_normalization([1, 2, 3, 4, 5])
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> min_max_normalization([10, 20, 30])
        array([0. , 0.5, 1. ])
        >>> min_max_normalization([5])
        array([0.])
        >>> min_max_normalization([5, 5, 5])
        array([0., 0., 0.])
    """
    data_array = np.array(data, dtype=float)

    if len(data_array) == 0:
        return np.array([])

    min_val = np.min(data_array)
    max_val = np.max(data_array)

    # Handle case where all values are the same
    if max_val == min_val:
        return np.zeros_like(data_array)

    return (data_array - min_val) / (max_val - min_val)


def decimal_scaling_normalization(data: list[float] | np.ndarray) -> np.ndarray:
    """
    Normalize data using Decimal Scaling normalization.

    Moves the decimal point of values to scale them to [-1, 1] range.
    Formula: X_normalized = X / (10^d) where d is the smallest integer
    such that max(|X_normalized|) < 1

    Args:
        data: Input data as list or numpy array

    Returns:
        Normalized data

    Examples:
        >>> decimal_scaling_normalization([100, 200, 300])
        array([0.1, 0.2, 0.3])
        >>> decimal_scaling_normalization([1, 2, 3])
        array([0.1, 0.2, 0.3])
        >>> decimal_scaling_normalization([-50, 0, 50])
        array([-0.5,  0. ,  0.5])
    """
    data_array = np.array(data, dtype=float)

    if len(data_array) == 0:
        return np.array([])

    max_abs = np.max(np.abs(data_array))

    if max_abs == 0:
        return data_array

    # Find the number of digits
    d = int(np.ceil(np.log10(max_abs + 1)))

    return data_array / (10**d)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    data = [10, 20, 30, 40, 50]
    print(f"Original data: {data}")
    print(f"Min-Max normalized: {min_max_normalization(data)}")
    print(f"Decimal scaling normalized: {decimal_scaling_normalization(data)}")
