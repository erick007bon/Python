"""
Time Series Analysis

Methods for analyzing and processing time series data.
"""

import numpy as np


def simple_moving_average(
    data: list[float] | np.ndarray, window_size: int
) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA).

    SMA smooths data by averaging values within a sliding window.

    Args:
        data: Time series data
        window_size: Size of the moving window

    Returns:
        Moving average values

    Examples:
        >>> simple_moving_average([1, 2, 3, 4, 5], 3)
        array([2., 3., 4.])
        >>> simple_moving_average([10, 20, 30, 40, 50], 2)
        array([15., 25., 35., 45.])
    """
    data_array = np.array(data, dtype=float)

    if window_size <= 0:
        msg = "Window size must be positive"
        raise ValueError(msg)

    if window_size > len(data_array):
        msg = "Window size cannot exceed data length"
        raise ValueError(msg)

    # Use convolution for efficient moving average
    weights = np.ones(window_size) / window_size
    return np.convolve(data_array, weights, mode="valid")


def exponential_moving_average(
    data: list[float] | np.ndarray, alpha: float = 0.3
) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent observations.
    Formula: EMA_t = alpha * X_t + (1 - alpha) * EMA_{t-1}

    Args:
        data: Time series data
        alpha: Smoothing factor (0 < alpha <= 1), higher = more weight to recent

    Returns:
        Exponential moving average values

    Examples:
        >>> result = exponential_moving_average([1, 2, 3, 4, 5], alpha=0.5)
        >>> len(result)
        5
        >>> result[0]
        1.0
    """
    data_array = np.array(data, dtype=float)

    if not 0 < alpha <= 1:
        msg = "Alpha must be between 0 and 1"
        raise ValueError(msg)

    ema = np.zeros_like(data_array)
    ema[0] = data_array[0]

    for i in range(1, len(data_array)):
        ema[i] = alpha * data_array[i] + (1 - alpha) * ema[i - 1]

    return ema


def weighted_moving_average(
    data: list[float] | np.ndarray, weights: list[float] | np.ndarray
) -> np.ndarray:
    """
    Calculate Weighted Moving Average (WMA).

    Applies custom weights to values in the window.

    Args:
        data: Time series data
        weights: Weights for each position in window (must sum to 1)

    Returns:
        Weighted moving average values

    Examples:
        >>> weighted_moving_average([1, 2, 3, 4, 5], [0.5, 0.3, 0.2])
        array([2.2, 3.2, 4.2])
        >>> weighted_moving_average([10, 20, 30], [0.6, 0.4])
        array([16., 26.])
    """
    data_array = np.array(data, dtype=float)
    weights_array = np.array(weights, dtype=float)

    if not np.isclose(np.sum(weights_array), 1.0):
        msg = "Weights must sum to 1"
        raise ValueError(msg)

    window_size = len(weights_array)

    if window_size > len(data_array):
        msg = "Window size cannot exceed data length"
        raise ValueError(msg)

    return np.convolve(data_array, weights_array[::-1], mode="valid")


def calculate_trend(data: list[float] | np.ndarray) -> tuple[float, float]:
    """
    Calculate linear trend (slope and intercept) using least squares.

    Args:
        data: Time series data

    Returns:
        Tuple of (slope, intercept)

    Examples:
        >>> slope, intercept = calculate_trend([1, 2, 3, 4, 5])
        >>> round(slope, 1)
        1.0
        >>> round(intercept, 1)
        1.0
        >>> slope, intercept = calculate_trend([10, 20, 30, 40])
        >>> round(slope, 1)
        10.0
    """
    data_array = np.array(data, dtype=float)
    n = len(data_array)
    x = np.arange(n)

    # Calculate slope and intercept using least squares
    slope = (n * np.sum(x * data_array) - np.sum(x) * np.sum(data_array)) / (
        n * np.sum(x**2) - np.sum(x) ** 2
    )

    intercept = (np.sum(data_array) - slope * np.sum(x)) / n

    return float(slope), float(intercept)


def detrend(data: list[float] | np.ndarray) -> np.ndarray:
    """
    Remove linear trend from time series data.

    Args:
        data: Time series data

    Returns:
        Detrended data

    Examples:
        >>> result = detrend([1, 2, 3, 4, 5])
        >>> np.allclose(result, [0., 0., 0., 0., 0.], atol=1e-10)
        True
        >>> result = detrend([1, 3, 2, 4, 3])
        >>> round(result[0], 2)
        -0.4
    """
    data_array = np.array(data, dtype=float)
    n = len(data_array)
    x = np.arange(n)

    slope, intercept = calculate_trend(data_array)
    trend = slope * x + intercept

    return data_array - trend


def seasonal_decomposition_simple(
    data: list[float] | np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple seasonal decomposition (additive model).

    Decomposes time series into trend, seasonal, and residual components.
    Model: data = trend + seasonal + residual

    Args:
        data: Time series data
        period: Length of seasonal period

    Returns:
        Tuple of (trend, seasonal, residual)

    Examples:
        >>> data = [1, 2, 1, 2, 1, 2, 1, 2]
        >>> trend, seasonal, residual = seasonal_decomposition_simple(data, period=2)
        >>> len(trend) == len(data)
        True
    """
    data_array = np.array(data, dtype=float)
    n = len(data_array)

    if period <= 0 or period > n:
        msg = "Period must be positive and not exceed data length"
        raise ValueError(msg)

    # Calculate trend using moving average
    if n >= period:
        trend = simple_moving_average(data_array, period)
        # Pad trend to match original length
        pad_size = n - len(trend)
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        trend = np.pad(trend, (pad_left, pad_right), mode="edge")
    else:
        trend = np.full(n, np.mean(data_array))

    # Calculate seasonal component
    detrended = data_array - trend
    seasonal = np.zeros(n)

    for i in range(period):
        indices = np.arange(i, n, period)
        seasonal[indices] = np.mean(detrended[indices])

    # Calculate residual
    residual = data_array - trend - seasonal

    return trend, seasonal, residual


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    data = [10, 12, 15, 14, 18, 20, 22, 25, 28, 30]
    print(f"Original data: {data}")

    print("\nSimple Moving Average (window=3):")
    sma = simple_moving_average(data, 3)
    print(sma)

    print("\nExponential Moving Average (alpha=0.3):")
    ema = exponential_moving_average(data, alpha=0.3)
    print(ema)

    print("\nTrend Analysis:")
    slope, intercept = calculate_trend(data)
    print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")

    print("\nDetrended Data:")
    detrended = detrend(data)
    print(detrended)
