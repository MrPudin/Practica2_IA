from __future__ import annotations
import re
import argparse
from utils import read_sms, split_observations_and_labels
from random import Random


def tokenize_sms(message):
    """YOUR CODE HERE"""
    """Tokeniza un mensaje SMS en palabras en minúsculas"""
    sms_tokens = re.findall(r"\b\w+\b", message.lower())
    return sms_tokens



class MultinomialNaiveBayesClassifier:
    def __init__(self, assumed_probability=1):
        self.assumed_probability = assumed_probability
        self.vocabulary = set()
        self.class_counts = {}
        self.word_counts = {}

    def fit(self, observations, labels):
        """YOUR CODE HERE"""
        # Initialize class counts
        for label in labels:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

        # Build vocabulary and word counts
        for obs, label in zip(observations, labels):
            for word in obs:
                self.vocabulary.add(word)
                if label not in self.word_counts:
                    self.word_counts[label] = {}
                self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1

        return self

    def predict(self, observations):
        """YOUR CODE HERE"""
        return observations
    
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
    print(f"Dataset loaded: {len(messages)} messages")
    print(f"Classes: {set(labels)}\n")

    # Tokenize the messages
    """YOUR CODE HERE"""
    tokenized_messages = [tokenize_sms(message) for message in messages]


    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    """YOUR CODE HERE"""
    indices = list(range(len(tokenized_messages)))
    rng.shuffle(indices)
    
    split_point = int(len(tokenized_messages) * (1 - args.test_ratio))
    
    train_obs = [tokenized_messages[i] for i in indices[:split_point]]
    train_labels = [labels[i] for i in indices[:split_point]]
    test_obs = [tokenized_messages[i] for i in indices[split_point:]]
    test_labels = [labels[i] for i in indices[split_point:]]
    
    print(f"Training set: {len(train_obs)} messages")
    print(f"Test set: {len(test_obs)} messages\n")

    # Instantiate the decision tree classifier
    mnb = MultinomialNaiveBayesClassifier(
        assumed_probability=args.assumed_probability
    )

    # Train the classifier using the training data
    """YOUR CODE HERE"""
    print("Training the Multinomial Naive Bayes classifier...")
    mnb.fit(train_obs, train_labels)
    print("Training completed.\n")

    print(f"Vocabulary size: {len(mnb.vocabulary)}")
    print(f"Class distribution: {mnb.class_counts}\n")

    # Predict over the test set
    """YOUR CODE HERE"""
    predictions = mnb.predict(test_obs)

    # Evaluate these predictions using the accuracy score and print the information
    """YOUR CODE HERE"""
    train_accuracy = mnb.score(train_obs, train_labels)
    test_accuracy = mnb.score(test_obs, test_labels)

    print(f"Training set accuracy: {train_accuracy:.2f}")
    print(f"Test set accuracy: {test_accuracy:.2f}")


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
