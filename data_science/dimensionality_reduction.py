"""
Dimensionality Reduction Techniques

Methods for reducing the number of features while preserving information.
"""

import numpy as np


def pca(
    data: list[list[float]] | np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    PCA transforms data to a new coordinate system where the greatest
    variance lies on the first coordinate (first principal component),
    the second greatest variance on the second coordinate, and so on.

    Args:
        data: 2D array where rows are samples and columns are features
        n_components: Number of principal components to keep

    Returns:
        Tuple of (transformed data, principal components, explained variance ratio)

    Examples:
        >>> data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> transformed, components, variance = pca(data, n_components=1)
        >>> transformed.shape
        (4, 1)
        >>> components.shape
        (1, 2)
    """
    data_array = np.array(data, dtype=float)

    if data_array.ndim != 2:
        msg = "Data must be 2-dimensional"
        raise ValueError(msg)

    if n_components > min(data_array.shape):
        msg = "n_components cannot exceed min(n_samples, n_features)"
        raise ValueError(msg)

    # Center the data (subtract mean)
    mean = np.mean(data_array, axis=0)
    centered_data = data_array - mean

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_data.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    principal_components = eigenvectors[:, :n_components].T

    # Transform data
    transformed_data = np.dot(centered_data, principal_components.T)

    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance

    return transformed_data, principal_components, explained_variance_ratio


def linear_discriminant_analysis(
    data: list[list[float]] | np.ndarray,
    labels: list[int] | np.ndarray,
    n_components: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction.

    LDA finds linear combinations of features that best separate classes.
    Unlike PCA, LDA is supervised and uses class labels.

    Args:
        data: 2D array where rows are samples and columns are features
        labels: Class labels for each sample
        n_components: Number of discriminant components to keep

    Returns:
        Tuple of (transformed data, discriminant components)

    Examples:
        >>> data = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]]
        >>> labels = [0, 0, 0, 1, 1, 1]
        >>> transformed, components = linear_discriminant_analysis(data, labels, 1)
        >>> transformed.shape
        (6, 1)
    """
    data_array = np.array(data, dtype=float)
    labels_array = np.array(labels, dtype=int)

    if data_array.ndim != 2:
        msg = "Data must be 2-dimensional"
        raise ValueError(msg)

    if len(data_array) != len(labels_array):
        msg = "Data and labels must have same number of samples"
        raise ValueError(msg)

    n_classes = len(np.unique(labels_array))
    if n_components >= n_classes:
        msg = "n_components must be less than number of classes"
        raise ValueError(msg)

    # Calculate overall mean
    overall_mean = np.mean(data_array, axis=0)

    # Calculate within-class and between-class scatter matrices
    n_features = data_array.shape[1]
    sw = np.zeros((n_features, n_features))  # Within-class scatter
    sb = np.zeros((n_features, n_features))  # Between-class scatter

    for c in np.unique(labels_array):
        class_data = data_array[labels_array == c]
        class_mean = np.mean(class_data, axis=0)
        n_samples = len(class_data)

        # Within-class scatter
        centered = class_data - class_mean
        sw += np.dot(centered.T, centered)

        # Between-class scatter
        mean_diff = (class_mean - overall_mean).reshape(-1, 1)
        sb += n_samples * np.dot(mean_diff, mean_diff.T)

    # Solve generalized eigenvalue problem
    sw_inv = np.linalg.pinv(sw)
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(sw_inv, sb))

    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    discriminant_components = eigenvectors[:, :n_components].T

    # Transform data
    transformed_data = np.dot(data_array, discriminant_components.T)

    return transformed_data, discriminant_components


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage - PCA
    print("Principal Component Analysis (PCA):")
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    transformed, components, variance_ratio = pca(data, n_components=2)
    print(f"Original shape: {np.array(data).shape}")
    print(f"Transformed shape: {transformed.shape}")
    print(f"Explained variance ratio: {variance_ratio}")

    # Example usage - LDA
    print("\nLinear Discriminant Analysis (LDA):")
    data = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]]
    labels = [0, 0, 0, 1, 1, 1]
    transformed, components = linear_discriminant_analysis(data, labels, 1)
    print(f"Original shape: {np.array(data).shape}")
    print(f"Transformed shape: {transformed.shape}")
