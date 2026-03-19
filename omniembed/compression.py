import torch.nn as nn
from sklearn.decomposition import PCA

class PCACompressor:
    """
    Compresses embeddings to a smaller dimension using PCA truncation.
    """
    def __init__(self, target_dim):
        self.target_dim = target_dim
        self.pca = PCA(n_components=target_dim)

    def fit(self, embeddings):
        self.pca.fit(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        return explained

    def transform(self, embeddings):
        return self.pca.transform(embeddings)

    def fit_transform(self, embeddings):
        explained = self.fit(embeddings)
        return self.transform(embeddings), explained


class MatryoshkaProjection(nn.Module):
    """
    Supervised linear projectors for Matryoshka Representation Learning.
    Projects from the base dimension down to a list of smaller nested dimensions.
    """
    def __init__(self, in_dim, out_dims=[256, 512, 1024]):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(in_dim, d, bias=False) for d in out_dims
        ])
    
    def forward(self, x):
        """
        Returns a list of projections for each desired dimension.
        """
        return [proj(x) for proj in self.projectors]
