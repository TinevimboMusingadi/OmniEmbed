import numpy as np

class ModalityAligner:
    """
    Handles translation and normalization for eliminating modality gaps in embedding spaces.
    """
    def __init__(self):
        self.zca_matrix = None
        self.zca_means = None

    @staticmethod
    def center_embeddings(embeddings_dict):
        """Subtracts the per-modality centroid."""
        centered_dict = {}
        for modality, embs in embeddings_dict.items():
            centered_dict[modality] = embs - embs.mean(axis=0)
        return centered_dict

    @staticmethod
    def zscore_embeddings(embeddings_dict):
        """Applies zero-mean unit-variance scaling per modality."""
        normalized_dict = {}
        for modality, embs in embeddings_dict.items():
            mean = embs.mean(axis=0)
            std = embs.std(axis=0) + 1e-8
            normalized_dict[modality] = (embs - mean) / std
        return normalized_dict

    def fit_zca(self, embeddings_dict):
        """
        Fits ZCA Whitening matrix across all joint modality embeddings
        to decorrelate features.
        """
        all_embs = np.vstack(list(embeddings_dict.values()))
        self.zca_means = all_embs.mean(axis=0)
        
        centered_all = all_embs - self.zca_means
        
        # Compute covariance
        cov = np.cov(centered_all.T)
        U, S, Vt = np.linalg.svd(cov)
        
        # ZCA transform matrix
        self.zca_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-5)) @ U.T

    def transform_zca(self, embeddings_dict):
        """Applies fitted ZCA transformation to given embeddings."""
        if self.zca_matrix is None:
            raise RuntimeError("ZCA ModalityAligner not fitted. Call fit_zca first.")
            
        whitened_dict = {}
        for modality, embs in embeddings_dict.items():
            centered_embs = embs - self.zca_means
            whitened_dict[modality] = (centered_embs @ self.zca_matrix.T)
            
        return whitened_dict
