from __future__ import annotations

from math import log
from dataclasses import dataclass
from typing import Optional
import argparse
from random import Random
from itertools import combinations

from utils import read_csv, split_observations_and_labels


def gini(labels) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    results = _unique_counts(labels)
    imp = 1.0
    for _, count in results.items():
        prob = count / total
        imp -= prob**2
    return imp


def entropy(labels) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    results = _unique_counts(labels)
    imp = 0.0
    for _, count in results.items():
        prob = count / total
        imp -= prob * _log2(prob)
    return imp


class DecisionTreeClassifier:
    def __init__(self, scoref=gini, beta=0.0, prune_threshold=0.0):
        self.scoref = scoref
        self.beta = beta
        self.prune_threshold = prune_threshold
        self.tree_: Optional[Node] = None

    def fit(self, observations, labels):
        self._iterative_build_tree(observations, labels)
        self._prune_tree()
        return self

    def predict(self, observations):
        if self.tree_ is None:
            return []
        out = []
        for observation in observations:
            leaf = self.tree_.follow_tree(observation)
            # clase mayoritaria en la hoja
            label = max(leaf.results.items(), key=lambda x: x[1])[0]
            out.append(label)
        return out

    def score(self, data, labels) -> float:
        if len(data) == 0:
            return 0.0
        predicted = self.predict(data)
        correct = sum(1 for pred, expected in zip(predicted, labels) if pred == expected)
        return correct / len(data)

    def _iterative_build_tree(self, observations, labels):
        # Raíz: si no hay datos, hoja vacía
        if len(observations) == 0:
            self.tree_ = Node.new_leaf([])
            return

        # Tareas en pila: (parent_node, side, obs, labels)
        # side: None para raíz, "true" o "false" para enganchar
        stack = [(None, None, observations, labels)]
        self.tree_ = None

        while stack:
            parent, side, obs, lab = stack.pop()

            # Casos base
            if len(obs) == 0:
                node = Node.new_leaf([])
            elif len(set(lab)) == 1:
                node = Node.new_leaf(lab)
            else:
                # Buscar mejor split
                num_features = len(obs[0])
                best_gain = float("-inf")
                best_criteria = None
                best_sets = None
                current_score = self.scoref(lab)

                for col in range(num_features):
                    values = _unique_values(obs, col)
                    if not values:
                        continue

                    sample_val = next(iter(values))

                    # Candidatos:
                    # - Numérico: umbral por cada valor único (simple)
                    # - Categórico: TODOS los subconjuntos no vacíos y no completos (∈)
                    if _is_numeric(sample_val):
                        candidates = values
                    else:
                        # Si hay 1 solo valor, no hay split posible
                        if len(values) <= 1:
                            continue
                        candidates = _categorical_subsets(values)

                    for value in candidates:
                        s1_obs, s1_lab, s2_obs, s2_lab = _divideset(obs, lab, col, value)
                        if len(s1_obs) == 0 or len(s2_obs) == 0:
                            continue

                        p = len(s1_obs) / len(obs)
                        gain = current_score - p * self.scoref(s1_lab) - (1 - p) * self.scoref(s2_lab)

                        if gain > best_gain:
                            best_gain = gain
                            best_criteria = (col, value)
                            best_sets = ((s1_obs, s1_lab), (s2_obs, s2_lab))

                # Criterio beta (min_gain): split solo si gain > beta
                if best_sets is not None and best_gain > self.beta:
                    node = Node.new_node(best_criteria[0], best_criteria[1], None, None)

                    # LIFO: meto false y luego true para construir true antes
                    stack.append((node, "false", best_sets[1][0], best_sets[1][1]))
                    stack.append((node, "true", best_sets[0][0], best_sets[0][1]))
                else:
                    node = Node.new_leaf(lab)

            # Enganchar
            if parent is None:
                self.tree_ = node
            else:
                if side == "true":
                    parent.true_branch = node
                else:
                    parent.false_branch = node

    def _prune_tree(self):
        if self.tree_ is None:
            return

        def results_to_labels(results_dict):
            lab = []
            for label, count in results_dict.items():
                lab.extend([label] * count)
            return lab

        def merge_results(a, b):
            merged = dict(a)
            for k, v in b.items():
                merged[k] = merged.get(k, 0) + v
            return merged

        def prune_rec(node: Node) -> Node:
            if node is None or node.is_leaf():
                return node

            node.true_branch = prune_rec(node.true_branch)
            node.false_branch = prune_rec(node.false_branch)

            # Solo intentamos fusionar si ambos hijos son hojas
            if node.true_branch.is_leaf() and node.false_branch.is_leaf():
                tr = node.true_branch.results or {}
                fr = node.false_branch.results or {}

                n_true = sum(tr.values())
                n_false = sum(fr.values())
                total = n_true + n_false
                if total == 0:
                    return node

                # Impureza antes: ponderada de hijos
                imp_before = (n_true / total) * self.scoref(results_to_labels(tr)) + \
                             (n_false / total) * self.scoref(results_to_labels(fr))

                # Impureza después: hoja fusionada
                merged = merge_results(tr, fr)
                imp_after = self.scoref(results_to_labels(merged))

                # Podar si no empeora "mucho"
                if (imp_after - imp_before) < self.prune_threshold:
                    return Node.new_leaf(results_to_labels(merged))

            return node

        self.tree_ = prune_rec(self.tree_)


@dataclass
class Node:
    column: Optional[int]
    value: Optional[int | float | str | frozenset]
    results: Optional[dict[int | float | str, int]]
    true_branch: Optional["Node"]
    false_branch: Optional["Node"]

    def is_leaf(self):
        return self.true_branch is None and self.false_branch is None

    @classmethod
    def new_node(cls, column, value, true_branch, false_branch):
        return cls(column, value, None, true_branch, false_branch)

    @classmethod
    def new_leaf(cls, labels):
        return cls(None, None, _unique_counts(labels), None, None)

    def print_tree(self, indent=""):
        if self.is_leaf():
            print(self.results)
        else:
            if _is_numeric(self.value):
                print(f"{self.column}: <= {self.value}?")
            elif isinstance(self.value, (set, frozenset)):
                print(f"{self.column}: in {set(self.value)}?")
            else:
                print(f"{self.column}: == {self.value}?")

            print(f"{indent}T->", end="")
            self.true_branch.print_tree(indent + " ")
            print(f"{indent}F->", end="")
            self.false_branch.print_tree(indent + " ")

    def follow_tree(self, observation):
        current = self
        while not current.is_leaf():
            query_fn = _get_query_fn(current.column, current.value)
            current = current.true_branch if query_fn(observation) else current.false_branch
        return current


###############################################
#             UTILITY FUNCTIONS               #
###############################################


def _unique_counts(values):
    results = {}
    for value in values:
        results[value] = results.get(value, 0) + 1
    return results


def _is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


def _get_query_fn(column, value):
    # Numérico: <=
    if _is_numeric(value):
        return lambda row: row[column] <= value
    # Subconjunto categórico: ∈
    if isinstance(value, (set, frozenset)):
        return lambda row: row[column] in value
    # Categórico simple: ==
    return lambda row: row[column] == value


def _unique_values(table, column_idx):
    values = set()
    for row in table:
        values.add(row[column_idx])
    return values


def _categorical_subsets(values):
    """
    Devuelve todos los subconjuntos no vacíos y no completos como frozenset.
    Si values = {A,B,C} -> {A},{B},{C},{A,B},{A,C},{B,C}
    """
    values = list(values)
    out = []
    for r in range(1, len(values)):  # excluye conjunto completo
        for comb in combinations(values, r):
            out.append(frozenset(comb))
    return out


def _log2(x):
    if x == 0:
        return 0.0
    return log(x) / log(2)


def _divideset(observations, labels, column, value):
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
    rng = Random(args.seed)

    # Cargar dataset (ignore_first=True: primera fila es cabecera)
    dataset = read_csv(args.dataset, ignore_first=True)
    observations, labels = split_observations_and_labels(dataset)

    # Split train/test
    indices = list(range(len(observations)))
    rng.shuffle(indices)

    split_point = int(len(observations) * (1 - args.test_ratio))

    train_obs = [observations[i] for i in indices[:split_point]]
    train_labels = [labels[i] for i in indices[:split_point]]
    test_obs = [observations[i] for i in indices[split_point:]]
    test_labels = [labels[i] for i in indices[split_point:]]

    print(f"Dataset loaded: {len(observations)} samples")
    print(f"Training set: {len(train_obs)} samples")
    print(f"Test set: {len(test_obs)} samples")
    print(f"Number of features: {len(observations[0]) if observations else 0}")
    print(f"Classes: {set(labels)}\n")

    # Instanciar clasificador
    dec_tree = DecisionTreeClassifier(
        scoref=gini if args.scoref == "gini" else entropy,
        beta=args.beta,
        prune_threshold=args.prune_threshold,
    )

    # Entrenar
    print("Tree Training:")
    dec_tree.fit(train_obs, train_labels)

    # Imprimir estructura
    print("Tree Structure:")
    dec_tree.tree_.print_tree()

    # Evaluación
    train_accuracy = dec_tree.score(train_obs, train_labels)
    test_accuracy = dec_tree.score(test_obs, test_labels)

    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to the CSV file containing the dataset.")
    parser.add_argument(
        "--scoref",
        type=str,
        choices=["gini", "entropy"],
        default="gini",
        help="Impurity measure criterion for the decision tree.",
    )
    parser.add_argument("--beta", type=float, default=0.0, help="Beta (min_gain) parameter.")
    parser.add_argument("--prune-threshold", type=float, default=0.0, help="Prune threshold.")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="Ratio for the test set split.")
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
