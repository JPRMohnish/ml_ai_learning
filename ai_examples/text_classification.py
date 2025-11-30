"""
Text Classification with Bag of Words
======================================
This module implements basic text classification using
bag-of-words features and naive Bayes.

Learning Resources:
- Natural Language Processing with Python (NLTK Book)
- Stanford NLP Course
"""

import re
import math
from collections import defaultdict


class SimpleTokenizer:
    """
    Simple text tokenizer that handles basic preprocessing.
    """

    def __init__(self, lowercase=True, min_length=2):
        """
        Initialize the tokenizer.

        Args:
            lowercase: Whether to convert text to lowercase
            min_length: Minimum token length to keep
        """
        self.lowercase = lowercase
        self.min_length = min_length

    def tokenize(self, text):
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()

        # Remove non-alphanumeric characters and split
        tokens = re.findall(r"\b[a-z]+\b", text)

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_length]

        return tokens


class BagOfWords:
    """
    Bag of Words vectorizer for text representation.
    """

    def __init__(self, max_features=None):
        """
        Initialize the vectorizer.

        Args:
            max_features: Maximum number of features to keep
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.tokenizer = SimpleTokenizer()

    def fit(self, documents):
        """
        Build vocabulary from documents.

        Args:
            documents: List of text documents

        Returns:
            self
        """
        word_counts = defaultdict(int)

        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            for token in set(tokens):
                word_counts[token] += 1

        # Sort by frequency and limit features
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        if self.max_features:
            sorted_words = sorted_words[: self.max_features]

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        return self

    def transform(self, documents):
        """
        Transform documents to bag-of-words vectors.

        Args:
            documents: List of text documents

        Returns:
            List of word count vectors
        """
        vectors = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            vector = [0] * vocab_size
            tokens = self.tokenizer.tokenize(doc)

            for token in tokens:
                if token in self.vocabulary:
                    vector[self.vocabulary[token]] += 1

            vectors.append(vector)

        return vectors

    def fit_transform(self, documents):
        """
        Fit and transform in one step.

        Args:
            documents: List of text documents

        Returns:
            List of word count vectors
        """
        self.fit(documents)
        return self.transform(documents)


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for text classification.

    This implementation demonstrates:
    - Bayes' theorem for classification
    - Laplace smoothing
    - Log probabilities for numerical stability
    """

    def __init__(self, alpha=1.0):
        """
        Initialize the classifier.

        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Args:
            X: Feature vectors (list of lists)
            y: Class labels

        Returns:
            self
        """
        self.classes = list(set(y))
        n_samples = len(y)
        n_features = len(X[0])

        # Compute class priors
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1

        for cls in self.classes:
            self.class_priors[cls] = class_counts[cls] / n_samples

        # Compute feature probabilities for each class
        self.feature_probs = {}
        for cls in self.classes:
            # Get samples for this class
            class_samples = [X[i] for i in range(n_samples) if y[i] == cls]

            # Sum feature counts across all samples
            feature_sums = [0] * n_features
            for sample in class_samples:
                for j in range(n_features):
                    feature_sums[j] += sample[j]

            total_count = sum(feature_sums)

            # Compute probabilities with Laplace smoothing
            self.feature_probs[cls] = [
                (count + self.alpha) / (total_count + self.alpha * n_features)
                for count in feature_sums
            ]

        return self

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X: Feature vectors

        Returns:
            List of predicted class labels
        """
        predictions = []

        for sample in X:
            best_class = None
            best_log_prob = float("-inf")

            for cls in self.classes:
                # Start with log prior
                log_prob = math.log(self.class_priors[cls])

                # Add log likelihoods
                for j, count in enumerate(sample):
                    if count > 0:
                        log_prob += count * math.log(self.feature_probs[cls][j])

                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_class = cls

            predictions.append(best_class)

        return predictions

    def score(self, X, y):
        """
        Compute classification accuracy.

        Args:
            X: Feature vectors
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)


def demo():
    """
    Demonstrate text classification on sample data.
    """
    # Sample training data - sentiment classification
    train_texts = [
        "I love this movie, it was fantastic and amazing",
        "Great film with wonderful acting and story",
        "This is the best movie I have ever seen",
        "Absolutely brilliant performance by the actors",
        "Terrible movie, waste of time and money",
        "I hated this film, it was boring and dull",
        "Worst movie ever, completely disappointing",
        "Awful acting and terrible plot, avoid this",
    ]
    train_labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
    ]

    # Test data
    test_texts = [
        "This movie was wonderful and entertaining",
        "I did not enjoy this boring film at all",
    ]
    test_labels = ["positive", "negative"]

    # Create bag of words features
    vectorizer = BagOfWords(max_features=100)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train classifier
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train, train_labels)

    # Evaluate
    train_accuracy = classifier.score(X_train, train_labels)
    test_predictions = classifier.predict(X_test)

    print("Text Classification Demo - Sentiment Analysis")
    print("=" * 50)
    print(f"Vocabulary size: {len(vectorizer.vocabulary)}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print("\nTest predictions:")
    for text, pred, true in zip(test_texts, test_predictions, test_labels):
        status = "✓" if pred == true else "✗"
        print(f"  {status} '{text[:40]}...' -> {pred} (expected: {true})")

    return classifier, vectorizer


if __name__ == "__main__":
    demo()
