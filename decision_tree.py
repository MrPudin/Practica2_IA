from __future__ import annotations
from math import log
from dataclasses import dataclass
from typing import Optional
import argparse
from utils import read_csv, split_observations_and_labels
from random import Random


def gini(labels) -> float:
    total = len(labels)
    results = _unique_counts(labels)
    imp = 1

    for label, count in results.items():
        prob = count / total
        imp -= prob**2

    return imp


def entropy(labels):
    total = len(labels)
    results = _unique_counts(labels)
    imp = 0

    for label, count in results.items():
        prob = count / total
        imp -= prob * _log2(prob)


class DecisionTreeClassifier:
    def __init__(self, scoref=gini, beta=0, prune_threshold=0):
        self.scoref = scoref
        self.beta = beta
        self.prune_threshold = prune_threshold

    def fit(self, observations, labels):
        self._iterative_build_tree(observations, labels)
        self._prune_tree()
        return self

    def predict(self, observations):
        labels = []
        for observation in observations:
            leaf = self.tree_.follow_tree(observation)
            label = max(leaf.results.items(), key=lambda x: x[1])[0]
            labels.append(label)
        return labels

    def score(self, data, labels) -> float:
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)

    def _iterative_build_tree(self, observations, labels):
        """YOUR CODE HERE"""
        self.tree_ = None
        raise NotImplementedError("TODO")

    def _prune_tree(self):
        """YOUR CODE HERE"""
        raise NotImplementedError("TODO")


@dataclass
class Node:
    column: Optional[int]
    value: Optional[int | float | str]
    results: Optional[dict[int | float | str, int]]
    true_branch: Optional[Node]
    false_branch: Optional[Node]

    def is_leaf(self):
        return self.true_branch is None

    @classmethod
    def new_node(cls, column, value, true_branch, false_branch):
        """Create a new instance of this class representing a decision node."""
        return cls(column, value, None, true_branch, false_branch)

    @classmethod
    def new_leaf(cls, labels):
        """Create a new instance of this class representing a leaf."""
        return cls(None, None, _unique_counts(labels), None, None)

    def print_tree(self, indent=""):
        """Prints to stdout a representation of the tree."""
        if self.is_leaf():
            print(self.results)
        else:
            # Print the criteria
            if _is_numeric(self.value):
                print(f"{self.column}: <= {self.value}?")
            else:
                print(f"{self.column}: {self.value}?")
            # Print the branches
            print(f"{indent}T->", end="")
            self.true_branch.print_tree(indent + " ")
            print(f"{indent}F->", end="")
            self.false_branch.print_tree(indent + " ")

    def follow_tree(self, observation):
        """
        Traverse the (sub)tree by answering the queries, until a leaf is reached.

        This method returns the leaf that this observation reaches.
        """
        current = self
        while not current.is_leaf():
            query_fn = _get_query_fn(current.column, current.value)
            current = (
                current.true_branch if query_fn(observation) else current.false_branch
            )

        return current


###############################################
#             UTILITY FUNCTIONS               #
###############################################


def _unique_counts(values):
    """Count how many times each value appears in `values`"""
    results = {}
    for value in values:
        if value not in results:
            results[value] = 1
        else:
            results[value] += 1
    return results


def _is_numeric(value):
    """Checks if a value is numeric (i.e. a float or an int)"""
    return isinstance(value, int) or isinstance(value, float)


def _get_query_fn(column, value):
    """
    Create a function that separates observations based on a query.
    The query can be:

    a) categorical: the created function returns true
       iff. the observation has the exact value in the column specified.
    b) continuous: the created function returns true
       iff. the observation has a value smaller or equal than the
       reference one in the column specified.

    Note: consider any column with a numeric value as continuous.
    """
    if _is_numeric(value):
        return lambda prot: prot[column] <= value
    else:
        return lambda prot: prot[column] == value


def _unique_values(table, column_idx):
    """Returns a set of the values in the columns of a table."""
    values = set()
    for row in table:
        values.add(row[column_idx])
    return values


def _log2(x):
    return log(x) / log(2)


def _divideset(observations, labels, column, value):
    """
    Divides a set on a specific column.
    Can handle numeric or categorical values
    """
    query_fn = _get_query_fn(column, value)

    observations1, labels1, observations2, labels2 = [], [], [], []

    for row, label in zip(observations, labels):
        if query_fn(row):
            observations1.append(row)
            labels1.append(label)
        else:
            observations2.append(row)
            labels2.append(label)

    return observations1, labels1, observations2, labels2


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    dataset = read_csv(args.dataset)
    observations, labels = split_observations_and_labels(dataset)

    # Split the dataset into training and test sets
    # NOTE: consider args.test_ratio and args.seed
    """YOUR CODE HERE"""

    # Instantiate the decision tree classifier
    dec_tree = DecisionTreeClassifier(
        scoref=args.scoref, beta=args.beta, prune_threshold=args.prune_threshold
    )

    # Train the decision tree using the training data
    """YOUR CODE HERE"""

    # Print the tree structure
    print("Tree Structure:")
    dec_tree.tree_.print_tree()

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
        "--scoref",
        type=str,
        choices=["gini", "entropy"],
        default="gini",
        help="Impurity measure criterion for the decision tree.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Value for the 'beta' parameter in the decision tree.",
    )
    parser.add_argument(
        "--prune-threshold", type=float, default=0.0, help="Prune threshold."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
