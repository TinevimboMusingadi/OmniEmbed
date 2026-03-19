import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_modality_umap(embeddings_dict, title="UMAP Projection", save_path=None, random_state=42):
    """
    Projects all embeddings to 2D using UMAP and visualizes modality overlap.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn must be installed to use plot_modality_umap")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=random_state)
    
    all_embs = []
    labels = []
    
    for modality, embs in embeddings_dict.items():
        all_embs.append(embs)
        labels.extend([modality] * len(embs))
        
    all_embs = np.vstack(all_embs)
    projection = reducer.fit_transform(all_embs)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=projection[:, 0], 
        y=projection[:, 1],
        hue=labels,
        palette="tab10",
        alpha=0.7
    )
    plt.title(title)
    plt.legend(title="Modality")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

def plot_similarity_heatmap(sim_matrix, labels_x, labels_y, title="Similarity Heatmap", save_path=None):
    """
    Visualizes pairwise similarities between items.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, cmap="viridis", xticklabels=labels_x, yticklabels=labels_y)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
