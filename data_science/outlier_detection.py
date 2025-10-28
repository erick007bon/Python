"""
Outlier Detection Methods

Techniques for identifying outliers (anomalous data points) in datasets.
"""

import numpy as np


def iqr_outlier_detection(
    data: list[float] | np.ndarray, threshold: float = 1.5
) -> tuple[list[int], list[float]]:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    An outlier is defined as a value that falls below Q1 - threshold*IQR
    or above Q3 + threshold*IQR. Common threshold is 1.5 for outliers
    and 3.0 for extreme outliers.

    Args:
        data: Input data
        threshold: IQR multiplier (default 1.5)

    Returns:
        Tuple of (outlier indices, outlier values)

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        >>> indices, values = iqr_outlier_detection(data)
        >>> values
        [100.0]
        >>> data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 200]
        >>> indices, values = iqr_outlier_detection(data)
        >>> values
        [200.0]
    """
    data_array = np.array(data, dtype=float)

    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outlier_mask = (data_array < lower_bound) | (data_array > upper_bound)
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = data_array[outlier_mask].tolist()

    return outlier_indices, outlier_values


def z_score_outlier_detection(
    data: list[float] | np.ndarray, threshold: float = 3.0
) -> tuple[list[int], list[float]]:
    """
    Detect outliers using the Z-score method.

    An outlier is defined as a value with |z-score| > threshold.
    Common threshold is 3.0 (3 standard deviations from mean).

    Args:
        data: Input data
        threshold: Z-score threshold (default 3.0)

    Returns:
        Tuple of (outlier indices, outlier values)

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        >>> indices, values = z_score_outlier_detection(data)
        >>> values
        [100.0]
        >>> data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 26]
        >>> indices, values = z_score_outlier_detection(data)
        >>> values
        []
    """
    data_array = np.array(data, dtype=float)

    mean = np.mean(data_array)
    std = np.std(data_array, ddof=0)

    if std == 0:
        return [], []

    z_scores = np.abs((data_array - mean) / std)
    outlier_mask = z_scores > threshold

    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = data_array[outlier_mask].tolist()

    return outlier_indices, outlier_values


def modified_z_score_outlier_detection(
    data: list[float] | np.ndarray, threshold: float = 3.5
) -> tuple[list[int], list[float]]:
    """
    Detect outliers using the Modified Z-score method (more robust).

    Uses median and MAD (Median Absolute Deviation) instead of mean and std.
    More robust to outliers than standard Z-score method.

    Modified Z-score = 0.6745 * (X - median) / MAD
    where MAD = median(|X - median(X)|)

    Args:
        data: Input data
        threshold: Modified Z-score threshold (default 3.5)

    Returns:
        Tuple of (outlier indices, outlier values)

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        >>> indices, values = modified_z_score_outlier_detection(data)
        >>> values
        [100.0]
    """
    data_array = np.array(data, dtype=float)

    median = np.median(data_array)
    mad = np.median(np.abs(data_array - median))

    if mad == 0:
        # Use mean absolute deviation if MAD is 0
        mad = np.mean(np.abs(data_array - median))
        if mad == 0:
            return [], []

    modified_z_scores = np.abs(0.6745 * (data_array - median) / mad)
    outlier_mask = modified_z_scores > threshold

    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = data_array[outlier_mask].tolist()

    return outlier_indices, outlier_values


def remove_outliers(
    data: list[float] | np.ndarray,
    method: str = "iqr",
    threshold: float | None = None,
) -> np.ndarray:
    """
    Remove outliers from data using specified method.

    Args:
        data: Input data
        method: Detection method ('iqr', 'zscore', or 'modified_zscore')
        threshold: Threshold value (uses default if None)

    Returns:
        Data with outliers removed

    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        >>> remove_outliers(data, method='iqr')
        array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> remove_outliers(data, method='zscore')
        array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    """
    data_array = np.array(data, dtype=float)

    if method == "iqr":
        threshold = threshold if threshold is not None else 1.5
        outlier_indices, _ = iqr_outlier_detection(data_array, threshold)
    elif method == "zscore":
        threshold = threshold if threshold is not None else 3.0
        outlier_indices, _ = z_score_outlier_detection(data_array, threshold)
    elif method == "modified_zscore":
        threshold = threshold if threshold is not None else 3.5
        outlier_indices, _ = modified_z_score_outlier_detection(data_array, threshold)
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    mask = np.ones(len(data_array), dtype=bool)
    mask[outlier_indices] = False

    return data_array[mask]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 26, 200, 250]
    print(f"Original data: {data}")

    print("\nIQR Method:")
    indices, values = iqr_outlier_detection(data)
    print(f"  Outlier indices: {indices}")
    print(f"  Outlier values: {values}")

    print("\nZ-Score Method:")
    indices, values = z_score_outlier_detection(data)
    print(f"  Outlier indices: {indices}")
    print(f"  Outlier values: {values}")

    print("\nModified Z-Score Method:")
    indices, values = modified_z_score_outlier_detection(data)
    print(f"  Outlier indices: {indices}")
    print(f"  Outlier values: {values}")

    print("\nData after removing outliers (IQR):")
    print(remove_outliers(data, method="iqr"))
