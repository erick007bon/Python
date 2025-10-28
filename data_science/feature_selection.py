"""
Feature Selection Techniques

Methods for selecting the most relevant features for machine learning models.
"""

import numpy as np


def variance_threshold_selector(
    data: list[list[float]] | np.ndarray, threshold: float = 0.0
) -> tuple[np.ndarray, list[int]]:
    """
    Select features based on variance threshold.

    Removes features with variance below the threshold.
    Low variance features are often less informative.

    Args:
        data: 2D array where rows are samples and columns are features
        threshold: Minimum variance threshold

    Returns:
        Tuple of (selected features data, selected feature indices)

    Examples:
        >>> data = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selected, indices = variance_threshold_selector(data, threshold=0.0)
        >>> indices
        [1, 2, 3]
        >>> data = [[1, 2, 3], [1, 4, 5], [1, 6, 7]]
        >>> selected, indices = variance_threshold_selector(data, threshold=0.5)
        >>> indices
        [1, 2]
    """
    data_array = np.array(data, dtype=float)

    if data_array.ndim != 2:
        msg = "Data must be 2-dimensional"
        raise ValueError(msg)

    # Calculate variance for each feature (column)
    variances = np.var(data_array, axis=0, ddof=0)

    # Select features with variance above threshold
    selected_indices = np.where(variances > threshold)[0].tolist()
    selected_data = data_array[:, selected_indices]

    return selected_data, selected_indices


def correlation_feature_selection(
    data: list[list[float]] | np.ndarray,
    target: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    """
    Select features based on correlation with target variable.

    Selects features with absolute correlation above threshold.

    Args:
        data: 2D array where rows are samples and columns are features
        target: Target variable values
        threshold: Minimum absolute correlation threshold

    Returns:
        Tuple of (selected features data, selected feature indices)

    Examples:
        >>> data = [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]]
        >>> target = [1, 2, 3, 4]
        >>> selected, indices = correlation_feature_selection(data, target, 0.9)
        >>> indices
        [0, 1, 2]
    """
    data_array = np.array(data, dtype=float)
    target_array = np.array(target, dtype=float)

    if data_array.ndim != 2:
        msg = "Data must be 2-dimensional"
        raise ValueError(msg)

    if len(data_array) != len(target_array):
        msg = "Data and target must have same number of samples"
        raise ValueError(msg)

    # Calculate correlation of each feature with target
    correlations = []
    for i in range(data_array.shape[1]):
        corr = np.corrcoef(data_array[:, i], target_array)[0, 1]
        correlations.append(abs(corr))

    correlations = np.array(correlations)

    # Select features with correlation above threshold
    selected_indices = np.where(correlations >= threshold)[0].tolist()
    selected_data = data_array[:, selected_indices]

    return selected_data, selected_indices


def mutual_information_score(
    x: list[float] | np.ndarray, y: list[float] | np.ndarray, bins: int = 10
) -> float:
    """
    Calculate mutual information between two variables (simplified version).

    Mutual information measures the mutual dependence between variables.
    This is a simplified discrete approximation using histograms.

    Args:
        x: First variable
        y: Second variable
        bins: Number of bins for discretization

    Returns:
        Mutual information score

    Examples:
        >>> x = [1, 2, 3, 4, 5]
        >>> y = [2, 4, 6, 8, 10]
        >>> score = mutual_information_score(x, y, bins=3)
        >>> score > 0
        True
    """
    x_array = np.array(x, dtype=float)
    y_array = np.array(y, dtype=float)

    # Create 2D histogram
    hist_2d, _, _ = np.histogram2d(x_array, y_array, bins=bins)

    # Calculate probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Calculate mutual information
    px_py = px[:, None] * py[None, :]

    # Avoid log(0)
    nonzero = pxy > 0
    mi = np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))

    return float(mi)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    print("Variance Threshold Selection:")
    data = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3], [0, 2, 2, 3]]
    print(f"Original data shape: {np.array(data).shape}")
    selected, indices = variance_threshold_selector(data, threshold=0.1)
    print(f"Selected features: {indices}")
    print(f"Selected data shape: {selected.shape}")

    print("\nCorrelation-based Selection:")
    data = [[1, 2, 10], [2, 4, 20], [3, 6, 15], [4, 8, 25]]
    target = [1, 2, 3, 4]
    selected, indices = correlation_feature_selection(data, target, threshold=0.8)
    print(f"Selected features: {indices}")
    print(f"Selected data shape: {selected.shape}")

    print("\nMutual Information:")
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    mi = mutual_information_score(x, y)
    print(f"Mutual information: {mi:.4f}")
