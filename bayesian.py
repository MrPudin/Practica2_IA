from __future__ import annotations
import argparse
from utils import read_sms, split_observations_and_labels
from random import Random


def tokenize_sms(message):
    """YOUR CODE HERE"""
    raise NotImplementedError("TODO")


class MultinomialNaiveBayesClassifier:
    def __init__(self, assumed_probability=1):
        self.assumed_probability = assumed_probability

    def fit(self, observations, labels):
        """YOUR CODE HERE"""
        return self

    def predict(self, observations):
        """YOUR CODE HERE"""
        raise NotImplementedError("TODO")

    def score(self, data, labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    messages, labels = read_sms(args.dataset)

    # Tokenize the messages
    """YOUR CODE HERE"""

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    """YOUR CODE HERE"""

    # Instantiate the decision tree classifier
    mnb = MultinomialNaiveBayesClassifier()

    # Train the classifier using the training data
    """YOUR CODE HERE"""

    # Predict over the test set
    """YOUR CODE HERE"""

    # Evaluate these predictions using the accuracy score and print the information
    """YOUR CODE HERE"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--assumed_probability",
        type=int,
        default=1,
        help="Value for the 'assumed_probability' parameter.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
