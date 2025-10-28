"""
Label Encoding for Categorical Data

Label encoding converts categorical labels into numerical values.
This is useful for machine learning algorithms that require numerical input.
"""


def label_encode(labels: list[str]) -> tuple[list[int], dict[str, int]]:
    """
    Encode categorical labels as integers.

    Args:
        labels: List of categorical labels

    Returns:
        Tuple of (encoded labels, label mapping dictionary)

    Examples:
        >>> label_encode(['cat', 'dog', 'cat', 'bird', 'dog'])
        ([0, 1, 0, 2, 1], {'cat': 0, 'dog': 1, 'bird': 2})
        >>> label_encode(['red', 'blue', 'green', 'red'])
        ([0, 1, 2, 0], {'red': 0, 'blue': 1, 'green': 2})
        >>> label_encode([])
        ([], {})
        >>> label_encode(['a'])
        ([0], {'a': 0})
    """
    if not labels:
        return [], {}

    # Create mapping of unique labels to integers
    unique_labels = []
    label_to_int = {}
    current_id = 0

    for label in labels:
        if label not in label_to_int:
            label_to_int[label] = current_id
            unique_labels.append(label)
            current_id += 1

    # Encode the labels
    encoded = [label_to_int[label] for label in labels]

    return encoded, label_to_int


def label_decode(
    encoded_labels: list[int], label_mapping: dict[str, int]
) -> list[str]:
    """
    Decode integer labels back to original categorical labels.

    Args:
        encoded_labels: List of encoded integer labels
        label_mapping: Dictionary mapping original labels to integers

    Returns:
        List of decoded categorical labels

    Examples:
        >>> mapping = {'cat': 0, 'dog': 1, 'bird': 2}
        >>> label_decode([0, 1, 0, 2, 1], mapping)
        ['cat', 'dog', 'cat', 'bird', 'dog']
        >>> label_decode([], mapping)
        []
    """
    if not encoded_labels:
        return []

    # Create reverse mapping
    int_to_label = {v: k for k, v in label_mapping.items()}

    # Decode the labels
    return [int_to_label[encoded] for encoded in encoded_labels]


def one_hot_encode(labels: list[str]) -> tuple[list[list[int]], dict[str, int]]:
    """
    One-hot encode categorical labels.

    Creates binary vectors where only one element is 1 (hot) and others are 0.

    Args:
        labels: List of categorical labels

    Returns:
        Tuple of (one-hot encoded vectors, label mapping dictionary)

    Examples:
        >>> one_hot_encode(['cat', 'dog', 'cat'])
        ([[1, 0], [0, 1], [1, 0]], {'cat': 0, 'dog': 1})
        >>> one_hot_encode(['a', 'b', 'c', 'a'])
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], {'a': 0, 'b': 1, 'c': 2})
        >>> one_hot_encode([])
        ([], {})
    """
    if not labels:
        return [], {}

    # Get label encoding first
    encoded, label_mapping = label_encode(labels)
    n_classes = len(label_mapping)

    # Create one-hot vectors
    one_hot = []
    for encoded_label in encoded:
        vector = [0] * n_classes
        vector[encoded_label] = 1
        one_hot.append(vector)

    return one_hot, label_mapping


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Example usage
    labels = ["cat", "dog", "bird", "cat", "dog", "bird", "fish"]
    print(f"Original labels: {labels}")

    encoded, mapping = label_encode(labels)
    print(f"Encoded labels: {encoded}")
    print(f"Label mapping: {mapping}")

    decoded = label_decode(encoded, mapping)
    print(f"Decoded labels: {decoded}")

    one_hot, mapping = one_hot_encode(labels)
    print(f"\nOne-hot encoded:")
    for label, vector in zip(labels, one_hot):
        print(f"  {label}: {vector}")
