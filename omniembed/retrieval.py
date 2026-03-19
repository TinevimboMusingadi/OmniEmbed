import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_recall_at_k(query_embeddings, target_embeddings, k_list=[1, 5, 10]):
    """
    Computes Recall@K for cross-modal retrieval tasks.
    Assumes the i-th query exactly corresponds to the i-th target.
    
    Args:
        query_embeddings (np.ndarray): Shape (N, D).
        target_embeddings (np.ndarray): Shape (N, D).
        k_list (list): K values for Recall@K.
        
    Returns:
        dict: A dictionary mapping 'Recall@K' to its score.
    """
    assert query_embeddings.shape[0] == target_embeddings.shape[0], "N of queries and targets must match."
    num_samples = query_embeddings.shape[0]
    
    # N x N pairwise similarities
    sim_matrix = cosine_similarity(query_embeddings, target_embeddings)
    
    # descending order indices
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    
    recalls = {}
    for k in k_list:
        correct = sum([1 for i in range(num_samples) if i in sorted_indices[i, :k]])
        recalls[f"Recall@{k}"] = correct / num_samples
        
    return recalls

def compute_modality_gap(emb_a, emb_b):
    """
    Measures the alignment gap between two sets of modality embeddings.
    
    Returns:
        dict: containing Euclidean 'l2_gap' and the 'cosine_gap' (1 - cos_sim).
    """
    mean_a = emb_a.mean(axis=0)
    mean_b = emb_b.mean(axis=0)
    
    gap_l2 = np.linalg.norm(mean_a - mean_b)
    gap_cosine = cosine_similarity([mean_a], [mean_b])[0, 0]
    
    return {"l2_gap": float(gap_l2), "cosine_gap": float(1 - gap_cosine)}

def alignment_score(x, y, alpha=2):
    """
    Alignment Metric from Wang & Isola (2020) - 'Understanding Contrastive Representation Learning'
    Expected distance between positive pairs. Lower is better.
    x and y should be torch Tensors on the same device.
    """
    import torch
    return (x - y).norm(dim=1).pow(alpha).mean().log().item()

def uniformity_score(x, t=2):
    """
    Uniformity Metric from Wang & Isola (2020)
    How well the embeddings are uniformly distributed on the hypersphere. Lower is better.
    x should be a torch Tensor.
    """
    import torch
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().item()
