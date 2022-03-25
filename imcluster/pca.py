from sklearn.decomposition import PCA

from rich.console import Console
console = Console()

from .io import ImclusterIO

def fit_pca(imcluster_io:ImclusterIO, feature_vectors, force:bool=False):
    if not imcluster_io.has_column('pca0') or not imcluster_io.has_column('pca1') or force:    
        console.print("Performing PCA")
        pca = PCA(n_components=2)
        feature_vectors_2D = pca.fit(feature_vectors).transform(feature_vectors)
    
        imcluster_io.save_column('pca0', feature_vectors_2D[:,0], autosave=False)
        imcluster_io.save_column('pca1', feature_vectors_2D[:,1], autosave=True)
    else:
        console.print("Using precomputed PCA")