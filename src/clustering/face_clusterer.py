from __future__ import annotations
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class FaceRecord:
    """
    Represents a single face record retrieved from the database or ingestion pipeline.

    Attributes:
        face_id (int): The unique database ID for this face entry.
        embedding (np.ndarray): The 512-D normalized vector from InsightFace.
        original_file_path (str): Path to the source image (useful for debugging).
    """
    face_id: int
    embedding: np.ndarray
    original_file_path: str

@dataclass
class ClusterResult:
    """
    Represents a grouped cluster of faces effectively identifying a unique person.

    Attributes:
        cluster_id (int): The label assigned by DBSCAN.
        member_ids (List[int]): List of face_ids belonging to this person.
        representative_face_id (int): The face_id closest to the mathematical center of the cluster.
        centroid (np.ndarray): The average embedding vector of this person (for future matching).
    """
    cluster_id: int
    member_ids: List[int]
    representative_face_id: int
    centroid: np.ndarray

class FaceClusterer:
    """
    Handles the unsupervised clustering of face embeddings to identify unique people
    without manual labeling. Uses DBSCAN with Cosine Similarity.
    """

    def __init__(self, eps: float = 0.4, min_samples: int = 3) -> None:
        """
        Initialize the DBSCAN clusterer.

        Args:
            eps (float): The maximum distance between two samples for one to be considered
                         in the neighborhood of the other.
                         - 0.4 is a standard starting point for ArcFace/InsightFace.
            min_samples (int): The number of samples required to form a dense region (a cluster).
        """
        self.eps = eps
        self.min_samples = min_samples
        # Initialize DBSCAN model with cosine similarity as metrics
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")

    def run(self, face_records: List[FaceRecord]) -> Dict[int, ClusterResult]:
        """
        Executes the clustering algorithm on the provided face records.

        Args:
            face_records (List[FaceRecord]): A list of FaceRecord objects.

        Returns:
            Dict[int, ClusterResult]: A dictionary mapping cluster_id to ClusterResult.
        """
        if not face_records:
            print("Warning: No face records provided to clusterer.")
            return {}

        # Unpack data for sklearn (shape: N * 512 (output of insightFace))
        embeddings = np.array([f.embedding for f in face_records])
        ids = [f.face_id for f in face_records]

        print(f"Clustering on {len(face_records)} faces.")

        # Fit the model
        self.model.fit(embeddings)
        labels = self.model.labels_

        # Process the results
        clusters: Dict[int, ClusterResult] = {}
        unique_labels = set(labels)

        # Filter out noise (-1)
        valid_labels = [l for l in unique_labels if l != -1]
        print(f"Found {len(valid_labels)} unique clusters.")

        for label in valid_labels:
            # Get all the indices that has $label
            indices = np.where(labels == label)[0]

            cluster_face_ids = [ids[i] for i in indices]
            cluster_embeddings = embeddings[indices]

            centroid = np.mean(cluster_embeddings, axis=0)

            distances = cosine_distances([centroid], cluster_embeddings)

            best_idx_local = np.argmin(distances)

            representative_face_id =  cluster_face_ids[best_idx_local]

            clusters[int(label)] = ClusterResult(
                int(label),
                cluster_face_ids,
                representative_face_id,
                centroid
            )

        return clusters
