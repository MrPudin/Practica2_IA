import argparse
from random import Random
from utils import read_csv


class KMeans:
    def __init__(self, k: int = 4, distance: str = "euclidean", rng=Random(123456)):
        self.k = k
        self.distance = distance
        self.rng = rng

    def fit(self, observations):
        """YOUR CODE HERE"""
        self.centroids_ = []
        self.distances_ = []
        self.X_assignments_ = []
        return self


###############################################
#                 CLI Code                    #
###############################################


def main(args):
    # Set the random generator
    rng = Random(args.seed)

    # Load the dataset
    dataset = read_csv(args.dataset)

    # Instantiate KMeans
    kmeans = KMeans(k=args.k, distance=args.distance, random_state=rng)

    # Train the clustering model
    """YOUR CODE HERE"""

    # Print some metrics
    print("Distances:", ...)
    print("Sum of distances:", ...)
    print("Centroid positions:", ...)
    print("Centroids assignments:", ...)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=str, help="Path to the CSV file containing the dataset."
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Value for the 'k' parameter of KMeans."
    )
    parser.add_argument(
        "--distance",
        type=str,
        choices=["euclidean", "squared-euclidean"],
        default="euclidean",
        help="Distance metric used by KMeans.",
    )
    parser.add_argument("--seed", type=int, default=123456, help="RNG Seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
