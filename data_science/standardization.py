"""
Data Standardization Techniques

Standardization transforms data to have zero mean and unit variance.
This is useful when features have different scales and you want to
center them around zero with a standard deviation of 1.
"""

import numpy as np


def z_score_standardization(data: list[float] | np.ndarray) -> np.ndarray:
    """
    Standardize data using Z-score standardization (zero mean, unit variance).

    Formula: X_standardized = (X - mean(X)) / std(X)

    Args:
        data: Input data as list or numpy array

    Returns:
        Standardized data with mean=0 and std=1

    Examples:
        >>> result = z_score_standardization([1, 2, 3, 4, 5])
        >>> np.allclose(result, [-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356])
        True
        >>> result = z_score_standardization([10, 10, 10])
        >>> np.allclose(result, [0., 0., 0.])
        True
        >>> z_score_standardization([5])
        array([0.])
    """
    data_array = np.array(data, dtype=float)

    if len(data_array) == 0:
        return np.array([])

    if len(data_array) == 1:
        return np.array([0.0])

    mean = np.mean(data_array)
    std = np.std(data_array, ddof=0)

    # Handle case where all values are the same
    if std == 0:
        return np.zeros_like(data_array)

    return (data_array - mean) / std


def robust_standardization(data: list[float] | np.ndarray) -> np.ndarray:
    """
    Standardize data using Robust Standardization (median and IQR).

    Uses median and interquartile range instead of mean and standard deviation.
    More robust to outliers than Z-score standardization.

    Formula: X_standardized = (X - median(X)) / IQR(X)
    where IQR = Q3 - Q1 (75th percentile - 25th percentile)

    Args:
        data: Input data as list or numpy array

    Returns:
        Robustly standardized data

    Examples:
        >>> result = robust_standardization([1, 2, 3, 4, 5])
        >>> np.allclose(result, [-1., -0.5, 0., 0.5, 1.])
        True
        >>> result = robust_standardization([1, 2, 3, 4, 100])
        >>> np.allclose(result, [-0.66666667, -0.33333333, 0., 0.33333333, 32.33333333])
        True
    """
    data_array = np.array(data, dtype=float)

    if len(data_array) == 0:
        return np.array([])

    if len(data_array) == 1:
        return np.array([0.0])

    median = np.median(data_array)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    # Handle case where IQR is zero
    if iqr == 0:
        return np.zeros_like(data_array)

    return (data_array - median) / iqr


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    data = [10, 20, 30, 40, 50]
    print(f"Original data: {data}")
    print(f"Z-score standardized: {z_score_standardization(data)}")
    print(f"Robust standardized: {robust_standardization(data)}")

    # Example with outliers
    data_with_outliers = [10, 20, 30, 40, 1000]
    print(f"\nData with outliers: {data_with_outliers}")
    print(f"Z-score standardized: {z_score_standardization(data_with_outliers)}")
    print(f"Robust standardized: {robust_standardization(data_with_outliers)}")
