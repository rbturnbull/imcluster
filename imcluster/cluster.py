from sklearn.cluster import DBSCAN, SpectralClustering

from rich.console import Console
console = Console()

from .io import ImclusterIO

def cluster(imcluster_io:ImclusterIO, feature_vectors, algorithm="SPECTRAL", force:bool=False):
    if not imcluster_io.has_column('cluster') or force:
        console.print("Clustering")

        algorithm = algorithm.upper()
        if algorithm == "SPECTRAL":
            clustering = SpectralClustering(n_clusters=2)
        elif algorithm == "DBSCAN":
            clustering = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise Exception(f"Cannot understand algorithm: {algorithm}")

        clustering.fit(feature_vectors)

        imcluster_io.save_column('cluster', clustering.labels_)
    else:
        console.print("Using precomputed clusters")
