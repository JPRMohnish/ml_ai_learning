"""
Linear Regression from Scratch
==============================
This module implements linear regression using gradient descent,
demonstrating core ML concepts.

Learning Resources:
- Andrew Ng's Machine Learning Course
- Stanford CS229 Lecture Notes
"""

import numpy as np


class LinearRegression:
    """
    Simple Linear Regression using Gradient Descent.

    This implementation demonstrates:
    - Gradient descent optimization
    - Mean squared error loss function
    - Feature normalization

    Attributes:
        learning_rate: Step size for gradient descent
        n_iterations: Number of training iterations
        weights: Model parameters (learned)
        bias: Intercept term (learned)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Linear Regression model.

        Args:
            learning_rate: Step size for gradient descent updates
            n_iterations: Number of iterations to run gradient descent
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {"loss": []}

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self: The trained model instance
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass: compute predictions
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss
            loss = self._compute_loss(y, y_predicted)
            self.history["loss"].append(loss)

        return self

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias

    def _compute_loss(self, y_true, y_pred):
        """
        Compute Mean Squared Error loss.

        Args:
            y_true: Actual target values
            y_pred: Predicted values

        Returns:
            Mean squared error value
        """
        return np.mean((y_true - y_pred) ** 2)

    def score(self, X, y):
        """
        Compute R-squared score.

        Args:
            X: Features of shape (n_samples, n_features)
            y: Target values

        Returns:
            R-squared coefficient of determination
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def demo():
    """
    Demonstrate linear regression on synthetic data.
    """
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100)

    # Split data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("Linear Regression Demo")
    print("=" * 40)
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")

    return model


if __name__ == "__main__":
    demo()
