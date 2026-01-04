from __future__ import annotations
import re
import argparse
from math import log
from utils import read_sms
from random import Random


def tokenize_sms(message):
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
        """Entrena el clasificador Naive Bayes Multinomial"""
        # Reinicializar estructuras
        self.class_counts = {}
        self.word_counts = {}
        self.vocabulary = set()
        
        # Procesar cada mensaje
        for tokens, label in zip(observations, labels):
            # Contar mensaje en esta clase
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            
            # Inicializar diccionario de palabras para esta clase si no existe
            if label not in self.word_counts:
                self.word_counts[label] = {}
            
            # Contar cada palabra del mensaje
            for word in tokens:
                self.vocabulary.add(word)
                self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1
        
        return self

    def predict(self, observations):
        """Predice la clase para cada observación usando Naive Bayes"""
        predictions = []
        
        # Pre-calcular valores constantes
        total_messages = sum(self.class_counts.values())
        vocab_size = len(self.vocabulary)
        
        # Calcular total de palabras por clase
        total_words_per_class = {}
        for label in self.class_counts:
            total_words_per_class[label] = sum(self.word_counts[label].values())
        
        # Clasificar cada mensaje
        for tokens in observations:
            class_scores = {}
            
            # Calcular score para cada clase
            for label in self.class_counts:
                # Log P(clase)
                prior_prob = self.class_counts[label] / total_messages
                score = log(prior_prob)
                
                # Sumar log P(palabra|clase) para cada palabra
                for word in tokens:
                    word_count = self.word_counts[label].get(word, 0)
                    total_words = total_words_per_class[label]
                    
                    # Laplace smoothing
                    word_prob = (word_count + self.assumed_probability) / \
                               (total_words + self.assumed_probability * vocab_size)
                    
                    score += log(word_prob)
                
                class_scores[label] = score
            
            # Elegir clase con mayor score
            predicted_label = max(class_scores, key=class_scores.get)
            predictions.append(predicted_label)
        
        return predictions
    
    def score(self, data, labels) -> float:
        """Calcula la precisión del clasificador"""
        predicted = self.predict(data)
        correct = sum(
            1 if pred == expected else 0 for pred, expected in zip(predicted, labels)
        )
        return correct / len(data)


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Generador aleatorio
    rng = Random(args.seed)

    # 1. Cargar dataset
    messages, labels = read_sms(args.dataset)
    print(f"Dataset loaded: {len(messages)} messages")
    print(f"Classes: {set(labels)}\n")

    # 2. Tokenizar mensajes
    tokenized_messages = [tokenize_sms(message) for message in messages]

    # 3. Dividir en train/test
    indices = list(range(len(tokenized_messages)))
    rng.shuffle(indices)
    
    split_point = int(len(tokenized_messages) * (1 - args.test_ratio))
    
    train_obs = [tokenized_messages[i] for i in indices[:split_point]]
    train_labels = [labels[i] for i in indices[:split_point]]
    test_obs = [tokenized_messages[i] for i in indices[split_point:]]
    test_labels = [labels[i] for i in indices[split_point:]]
    
    print(f"Training set: {len(train_obs)} messages")
    print(f"Test set: {len(test_obs)} messages\n")

    # 4. Crear clasificador
    mnb = MultinomialNaiveBayesClassifier(
        assumed_probability=args.assumed_probability
    )

    # 5. Entrenar
    print("Training the Multinomial Naive Bayes classifier...")
    mnb.fit(train_obs, train_labels)
    print("Training completed.\n")

    # Info del modelo
    print(f"Vocabulary size: {len(mnb.vocabulary)}")
    print(f"Class distribution: {mnb.class_counts}\n")

    # 6. Predecir
    predictions = mnb.predict(test_obs)

    # 7. Evaluar
    train_accuracy = mnb.score(train_obs, train_labels)
    test_accuracy = mnb.score(test_obs, test_labels)

    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the SMS dataset file."
    )
    parser.add_argument(
        "--assumed-probability",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter (default: 1.0).",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.3, help="Ratio for the test set split."
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)