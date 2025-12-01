# ML/AI Learning Repository

A hands-on repository for learning Machine Learning and Artificial Intelligence concepts from scratch using Python.

## Project Structure

```
ml_ai_learning/
├── ml_basics/           # Core ML implementations
│   ├── linear_regression.py    # Linear regression with gradient descent
│   └── neural_network.py       # Feedforward neural network
├── ai_examples/         # AI and NLP examples
│   └── text_classification.py  # Naive Bayes text classifier
├── utils/               # Utility functions
│   └── data_utils.py           # Data processing helpers
├── requirements.txt     # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JPRMohnish/ml_ai_learning.git
cd ml_ai_learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Linear Regression
```python
from ml_basics.linear_regression import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y):.4f}")
```

### Neural Network (XOR Problem)
```python
from ml_basics.neural_network import NeuralNetwork
import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Train network
model = NeuralNetwork(hidden_size=4, learning_rate=0.5, n_iterations=5000)
model.fit(X, y)

print(f"Predictions: {model.predict(X)}")
print(f"Accuracy: {model.score(X, y):.2%}")
```

### Text Classification
```python
from ai_examples.text_classification import BagOfWords, NaiveBayesClassifier

# Sample data
texts = ["I love this", "This is great", "I hate this", "This is terrible"]
labels = ["positive", "positive", "negative", "negative"]

# Train classifier
vectorizer = BagOfWords()
X = vectorizer.fit_transform(texts)
classifier = NaiveBayesClassifier()
classifier.fit(X, labels)

# Predict
test_text = ["I really love it"]
test_X = vectorizer.transform(test_text)
print(classifier.predict(test_X))
```

## Running Demos

Each module includes a demo function:

```bash
# Linear Regression demo
python -m ml_basics.linear_regression

# Neural Network demo
python -m ml_basics.neural_network

# Text Classification demo
python -m ai_examples.text_classification
```

## Learning Resources

### Machine Learning
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Stanford CS229 Lecture Notes](https://cs229.stanford.edu/notes2022fall/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

### Deep Learning
- [Deep Learning by Ian Goodfellow](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Natural Language Processing
- [Natural Language Processing with Python (NLTK Book)](https://www.nltk.org/book/)
- [Stanford NLP Course](https://web.stanford.edu/class/cs224n/)
- [Hugging Face Course](https://huggingface.co/course)

## Contributing

Feel free to add new implementations, improve existing code, or add more learning resources!

## License

This project is for educational purposes.
