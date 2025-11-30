"""
Data Processing Utilities
=========================
Common utilities for data preprocessing and manipulation.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into train and test subsets.

    Args:
        X: Features array
        y: Labels array
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
    y_arr = np.array(y) if not isinstance(y, np.ndarray) else y

    return X_arr[train_indices], X_arr[test_indices], y_arr[train_indices], y_arr[test_indices]


def normalize(X, axis=0):
    """
    Normalize features to zero mean and unit variance.

    Args:
        X: Features array of shape (n_samples, n_features)
        axis: Axis along which to normalize

    Returns:
        Normalized array, mean, and std
    """
    X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
    mean = np.mean(X_arr, axis=axis)
    std = np.std(X_arr, axis=axis)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    return (X_arr - mean) / std, mean, std


def min_max_scale(X, feature_range=(0, 1)):
    """
    Scale features to a given range.

    Args:
        X: Features array
        feature_range: Tuple of (min, max) for scaling

    Returns:
        Scaled array
    """
    X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
    min_val = np.min(X_arr, axis=0)
    max_val = np.max(X_arr, axis=0)
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)  # Avoid division by zero

    X_scaled = (X_arr - min_val) / range_val
    return X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """
    Compute mean squared error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label values to include

    Returns:
        Confusion matrix as 2D numpy array
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n_labels = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    matrix = np.zeros((n_labels, n_labels), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            matrix[label_to_idx[true], label_to_idx[pred]] += 1

    return matrix
