"""
Statistical Analysis Functions

Common statistical measures and tests for data analysis.
"""

import numpy as np


def mean(data: list[float] | np.ndarray) -> float:
    """
    Calculate the arithmetic mean (average) of data.

    Args:
        data: Input data

    Returns:
        Mean value

    Examples:
        >>> mean([1, 2, 3, 4, 5])
        3.0
        >>> mean([10, 20, 30])
        20.0
        >>> mean([5])
        5.0
    """
    data_array = np.array(data, dtype=float)
    return float(np.mean(data_array))


def median(data: list[float] | np.ndarray) -> float:
    """
    Calculate the median (middle value) of data.

    Args:
        data: Input data

    Returns:
        Median value

    Examples:
        >>> median([1, 2, 3, 4, 5])
        3.0
        >>> median([1, 2, 3, 4])
        2.5
        >>> median([5])
        5.0
    """
    data_array = np.array(data, dtype=float)
    return float(np.median(data_array))


def mode(data: list[float] | np.ndarray) -> float:
    """
    Calculate the mode (most frequent value) of data.

    Args:
        data: Input data

    Returns:
        Mode value (returns first mode if multiple exist)

    Examples:
        >>> mode([1, 2, 2, 3, 4])
        2.0
        >>> mode([1, 1, 2, 2, 3])
        1.0
        >>> mode([5])
        5.0
    """
    data_array = np.array(data, dtype=float)
    values, counts = np.unique(data_array, return_counts=True)
    return float(values[np.argmax(counts)])


def variance(data: list[float] | np.ndarray, sample: bool = True) -> float:
    """
    Calculate the variance of data.

    Args:
        data: Input data
        sample: If True, calculate sample variance (n-1), else population variance (n)

    Returns:
        Variance value

    Examples:
        >>> variance([1, 2, 3, 4, 5])
        2.5
        >>> variance([1, 2, 3, 4, 5], sample=False)
        2.0
        >>> round(variance([10, 20, 30]), 2)
        100.0
    """
    data_array = np.array(data, dtype=float)
    ddof = 1 if sample else 0
    return float(np.var(data_array, ddof=ddof))


def standard_deviation(data: list[float] | np.ndarray, sample: bool = True) -> float:
    """
    Calculate the standard deviation of data.

    Args:
        data: Input data
        sample: If True, calculate sample std (n-1), else population std (n)

    Returns:
        Standard deviation value

    Examples:
        >>> round(standard_deviation([1, 2, 3, 4, 5]), 2)
        1.58
        >>> round(standard_deviation([10, 20, 30]), 2)
        10.0
    """
    data_array = np.array(data, dtype=float)
    ddof = 1 if sample else 0
    return float(np.std(data_array, ddof=ddof))


def quartiles(data: list[float] | np.ndarray) -> tuple[float, float, float]:
    """
    Calculate the quartiles (Q1, Q2, Q3) of data.

    Args:
        data: Input data

    Returns:
        Tuple of (Q1, Q2, Q3)

    Examples:
        >>> quartiles([1, 2, 3, 4, 5])
        (2.0, 3.0, 4.0)
        >>> quartiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        (3.25, 5.5, 7.75)
    """
    data_array = np.array(data, dtype=float)
    q1 = float(np.percentile(data_array, 25))
    q2 = float(np.percentile(data_array, 50))
    q3 = float(np.percentile(data_array, 75))
    return q1, q2, q3


def interquartile_range(data: list[float] | np.ndarray) -> float:
    """
    Calculate the interquartile range (IQR = Q3 - Q1).

    Args:
        data: Input data

    Returns:
        IQR value

    Examples:
        >>> interquartile_range([1, 2, 3, 4, 5])
        2.0
        >>> interquartile_range([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        4.5
    """
    q1, _, q3 = quartiles(data)
    return q3 - q1


def correlation_coefficient(
    x: list[float] | np.ndarray, y: list[float] | np.ndarray
) -> float:
    """
    Calculate Pearson correlation coefficient between two variables.

    Measures linear correlation between -1 (negative) and 1 (positive).

    Args:
        x: First variable data
        y: Second variable data

    Returns:
        Correlation coefficient

    Examples:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> correlation_coefficient(x, y)
        1.0
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [5, 4, 3, 2, 1]
        >>> correlation_coefficient(x, y)
        -1.0
    """
    x_array = np.array(x, dtype=float)
    y_array = np.array(y, dtype=float)

    if len(x_array) != len(y_array):
        msg = "Arrays must have the same length"
        raise ValueError(msg)

    return float(np.corrcoef(x_array, y_array)[0, 1])


def covariance(
    x: list[float] | np.ndarray, y: list[float] | np.ndarray
) -> float:
    """
    Calculate covariance between two variables.

    Args:
        x: First variable data
        y: Second variable data

    Returns:
        Covariance value

    Examples:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> covariance(x, y)
        5.0
    """
    x_array = np.array(x, dtype=float)
    y_array = np.array(y, dtype=float)

    if len(x_array) != len(y_array):
        msg = "Arrays must have the same length"
        raise ValueError(msg)

    return float(np.cov(x_array, y_array)[0, 1])


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    data = [12, 15, 18, 20, 22, 25, 28, 30, 35, 40]
    print(f"Data: {data}")
    print(f"Mean: {mean(data)}")
    print(f"Median: {median(data)}")
    print(f"Variance: {variance(data)}")
    print(f"Standard Deviation: {standard_deviation(data)}")
    print(f"Quartiles: {quartiles(data)}")
    print(f"IQR: {interquartile_range(data)}")

    # Correlation example
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    print(f"\nX: {x}")
    print(f"Y: {y}")
    print(f"Correlation: {correlation_coefficient(x, y)}")
    print(f"Covariance: {covariance(x, y)}")
