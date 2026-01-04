from __future__ import annotations
import argparse
import re # Standard library for regular expressions
from utils import read_sms, split_observations_and_labels
from random import Random
from collections import defaultdict, Counter


def tokenize_sms(message):
    # In order to homogenize data
    message = message.lower()
    # Tokenize words + ! and ? : free!!!? ---> ['free', '!', '!', '!', '?']
    tokens = re.findall(r'\w+|[!?]', message)
    return tokens


class MultinomialNaiveBayesClassifier:
    def __init__(self, assumed_probability=1):
        self.assumed_probability = assumed_probability
        self.class_doc_counts = defaultdict(int)          # Number of docs per class
        self.class_word_counts = defaultdict(Counter)     # Number of times that a word appears per class
        self.vocabulary = set()                           # All the words without rep
        self.class_total_words = defaultdict(int)         # Total words per class
        self.total_docs = 0                               # Total docs

    def fit(self, observations, labels):
        for tokens, label in zip(observations, labels):
            self.class_doc_counts[label] += 1
            self.total_docs += 1
            self.class_word_counts[label].update(tokens)
            self.class_total_words[label] += len(tokens)
            self.vocabulary.update(tokens)
        return self

    def predict(self, observations):
        predictions = []
        vocab_size = len(self.vocabulary)
        
        for tokens in observations:
            class_probs = {}
            for label in self.class_doc_counts:
                # P(class)
                prior = self.class_doc_counts[label] / self.total_docs
                prob = prior  # empezamos con P(class)
                
                total_words = self.class_total_words[label]
                word_counts = self.class_word_counts[label] # Dictionary associated with the class "label" containing the words appeared on all the docs from that class and their counters associated
                
                # Multiply P(word|class) for each word
                for word in tokens:
                    count = word_counts.get(word, 0) # Get the word counter data
                    pwc = (count + self.assumed_probability) / (total_words + self.assumed_probability * vocab_size)
                    prob *= pwc
                
                class_probs[label] = prob
            
            # Choose the class with higher probability
            predicted_label = max(class_probs, key=class_probs.get)
            predictions.append(predicted_label)
        
        return predictions

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
    tokenized_messages = [tokenize_sms(m) for m in messages]

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    data = list(zip(tokenized_messages, labels)) # Result: tuples --->   [(['go', 'until', 'point', '!'], 'ham'),  (['ok', 'just', 'joking', 'with', 'u'], 'ham')]
    rng.shuffle(data)

    split_idx = int(len(data) * (1 - args.test_ratio))
    train_data, test_data = data[:split_idx], data[split_idx:]

    train_observations, train_labels = zip(*train_data)
    test_observations, test_labels = zip(*test_data)

    # Instantiate the decision tree classifier
    mnb = MultinomialNaiveBayesClassifier(assumed_probability=args.assumed_probability)

    # Train the classifier using the training data
    mnb.fit(train_labels, train_labels)

    # Predict over the test set
    predictions = mnb.predict(test_observations)

    # Evaluate these predictions using the accuracy score and print the information
    accuracy = mnb.score(test_observations, test_labels)
    print(f"Accuracy: {accuracy*100:.2f}%")


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
