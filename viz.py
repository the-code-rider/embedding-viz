from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import json
import numpy as np
import matplotlib.pyplot as plt


# import hdbscan


def reduce_embedding_dimension(embeddings, num_components=2, algo="PCA"):
    reduced_embeddings = None
    if algo == 'PCA':
        pca = PCA(n_components=num_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    elif algo == 'UMAP':
        reducer = umap.UMAP(n_components=num_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif algo == 'TSNE':
        tsne = TSNE(n_components=2, random_state=42)
        embeds = tsne.fit_transform(embeddings)


    return reduced_embeddings


def cluster_embeddings(embeddings, min_cluster_size=5):
    pass
    # todo : fix this
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    # clusters = clusterer.fit_predict(embeddings)
    # return clusters


def load_embeddings_from_file(file_path):
    # loads embedding from a file a np array
    emb_list = []
    with open(file_path) as f:
        for line in f:
            # import json
            emb = json.loads(line.replace("'", "\""))
            emb_list.append(emb)

    return np.array(emb_list)


def plot_scatter(embeddings, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5)
    if title:
        plt.title('2D Visualization of High-Dimensional Embeddings')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # Save the plot as a PNG file
    plt.savefig(filename, dpi=300)
