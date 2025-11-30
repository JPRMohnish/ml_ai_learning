"""
Neural Network from Scratch
============================
This module implements a simple feedforward neural network,
demonstrating core deep learning concepts.

Learning Resources:
- Deep Learning by Ian Goodfellow
- Neural Networks and Deep Learning by Michael Nielsen
"""

import numpy as np


class NeuralNetwork:
    """
    Simple Feedforward Neural Network with one hidden layer.

    This implementation demonstrates:
    - Forward propagation
    - Backpropagation
    - Activation functions (ReLU, Sigmoid)
    - Binary cross-entropy loss

    Attributes:
        hidden_size: Number of neurons in hidden layer
        learning_rate: Step size for gradient descent
        n_iterations: Number of training epochs
    """

    def __init__(self, hidden_size=10, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the neural network.

        Args:
            hidden_size: Number of neurons in the hidden layer
            learning_rate: Step size for gradient descent updates
            n_iterations: Number of training epochs
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.history = {"loss": []}

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        # Clip values to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        """Derivative of sigmoid function."""
        return a * (1 - a)

    def _relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        """Derivative of ReLU function."""
        return (z > 0).astype(float)

    def _initialize_weights(self, n_features):
        """
        Initialize weights using Xavier initialization.

        Args:
            n_features: Number of input features
        """
        np.random.seed(42)
        self.W1 = np.random.randn(n_features, self.hidden_size) * np.sqrt(
            2.0 / n_features
        )
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, 1))

    def _forward(self, X):
        """
        Forward propagation through the network.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Tuple of (output, cache) where cache contains intermediate values
        """
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)

        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)

        cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2, cache

    def _backward(self, X, y, cache):
        """
        Backpropagation to compute gradients.

        Args:
            X: Input features
            y: Target values
            cache: Intermediate values from forward pass

        Returns:
            Dictionary of gradients
        """
        n_samples = X.shape[0]
        a1 = cache["a1"]
        a2 = cache["a2"]
        z1 = cache["z1"]

        # Output layer gradients
        dz2 = a2 - y.reshape(-1, 1)
        dW2 = (1 / n_samples) * np.dot(a1.T, dz2)
        db2 = (1 / n_samples) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self._relu_derivative(z1)
        dW1 = (1 / n_samples) * np.dot(X.T, dz1)
        db1 = (1 / n_samples) * np.sum(dz1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def _compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.

        Args:
            y_true: Actual target values
            y_pred: Predicted probabilities

        Returns:
            Binary cross-entropy loss value
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def fit(self, X, y):
        """
        Train the neural network.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self: The trained model instance
        """
        n_features = X.shape[1]
        self._initialize_weights(n_features)

        for _ in range(self.n_iterations):
            # Forward pass
            y_pred, cache = self._forward(X)

            # Compute and store loss
            loss = self._compute_loss(y, y_pred.flatten())
            self.history["loss"].append(loss)

            # Backward pass
            gradients = self._backward(X, y, cache)

            # Update weights
            self.W1 -= self.learning_rate * gradients["dW1"]
            self.b1 -= self.learning_rate * gradients["db1"]
            self.W2 -= self.learning_rate * gradients["dW2"]
            self.b2 -= self.learning_rate * gradients["db2"]

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted probabilities of shape (n_samples,)
        """
        y_pred, _ = self._forward(X)
        return y_pred.flatten()

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Args:
            X: Features of shape (n_samples, n_features)
            threshold: Classification threshold

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, y):
        """
        Compute classification accuracy.

        Args:
            X: Features of shape (n_samples, n_features)
            y: True labels

        Returns:
            Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo():
    """
    Demonstrate neural network on XOR problem.
    """
    # XOR problem - non-linearly separable
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Train model
    model = NeuralNetwork(hidden_size=4, learning_rate=0.5, n_iterations=5000)
    model.fit(X, y)

    # Evaluate
    predictions = model.predict(X)
    accuracy = model.score(X, y)

    print("Neural Network Demo - XOR Problem")
    print("=" * 40)
    print(f"Input: {X.tolist()}")
    print(f"Expected: {y.tolist()}")
    print(f"Predicted: {predictions.tolist()}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Final loss: {model.history['loss'][-1]:.4f}")

    return model


if __name__ == "__main__":
    demo()
