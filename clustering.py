import argparse
import math
from copy import deepcopy
from random import Random
from utils import read_csv


class KMeans:
    def __init__(self, k: int = 4, distance: str = "euclidean", rng=Random(123456), n_restarts: int = 10):
        self.k = k
        self.distance = distance
        self.rng = rng
        self.n_restarts = n_restarts

    def _distance(self, x, y):
        """
        Calculate the disatance between two vectors
        """
        if self.distance == "euclidean":
            return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))
        elif self.distance == "squared-euclidean":
            return sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        else:
            raise ValueError(
                f"Unvalid distance parameter: '{self.distance}'. "
                "Use 'euclidean' or 'squared-euclidean'."
            )

    def fit(self, observations):
        n_samples = len(observations) # Number of points
        n_features = len(observations[0]) # Numbher of point dimensions

        best_sum = float("inf")
        best_centroids = None
        best_assignments = None
        best_distances = None

        for _ in range(self.n_restarts):
            # Initialize the centroids randomly from the dataset
            self.centroids_ = self.rng.sample(observations, self.k)

            # Kmeans algorythm iterations
            while True:
                old_centroids = deepcopy(self.centroids_)

                self.X_assignments_ = []
                self.distances_ = []

                # Point assignation to the closest centroid
                for x in observations:
                    distances_to_centroids = [
                        self._distance(x, c) for c in self.centroids_
                    ]

                    closest_centroid = distances_to_centroids.index(
                        min(distances_to_centroids)
                    )

                    self.X_assignments_.append(closest_centroid)
                    self.distances_.append(distances_to_centroids[closest_centroid])

                # Recalculate centroids
                new_centroids = []

                for j in range(self.k):
                    cluster_points = [
                        observations[i]
                        for i in range(n_samples)
                        if self.X_assignments_[i] == j
                    ]

                    # Empty cluster ---> Reinitialize centroid (randomly)
                    if not cluster_points:
                        new_centroids.append(
                            observations[self.rng.randrange(n_samples)]
                        )
                    else:
                        # Recalculate the centroid
                        centroid = [
                            sum(point[d] for point in cluster_points) / len(cluster_points)
                            for d in range(n_features)
                        ]
                        new_centroids.append(centroid)

                self.centroids_ = new_centroids

                # Convergence check
                if self.centroids_ == old_centroids:
                    break

            current_sum = sum(self.distances_)

            if current_sum < best_sum:
                best_sum = current_sum
                best_centroids = deepcopy(self.centroids_)
                best_assignments = self.X_assignments_[:]
                best_distances = self.distances_[:]

        self.centroids_ = best_centroids
        self.X_assignments_ = best_assignments
        self.distances_ = best_distances

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
    kmeans = KMeans(k=args.k, distance=args.distance, rng=rng, n_restarts=args.n_restarts)

    # Train the clustering model
    kmeans.fit(dataset)

    # Print some metrics
    print("Distances:", kmeans.distances_)
    print("Sum of distances:", sum(kmeans.distances_))
    print("Centroid positions:", kmeans.centroids_)
    print("Centroids assignments:", kmeans.X_assignments_)


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
    
    parser.add_argument(
        "--n-restarts", 
        type=int, 
        default=10, 
        help="Number of k-means executions with different inicializations."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
