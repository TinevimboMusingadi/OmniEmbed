import numpy as np
from omniembed.alignment import ModalityAligner

def test_alignment_methods():
    aligner = ModalityAligner()
    # Mock embeddings lists
    embs = {
        "text": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "image": np.array([[5.0, 6.0], [7.0, 8.0]])
    }
    
    # Test Centering
    centered = aligner.center_embeddings(embs)
    assert np.allclose(centered["text"].mean(axis=0), [0, 0])
    assert np.allclose(centered["image"].mean(axis=0), [0, 0])
    
    # Test Z-score
    zscored = aligner.zscore_embeddings(embs)
    # Each embedding array should now have unit variance
    assert np.allclose(zscored["text"].std(axis=0), [1, 1], atol=1e-3)
    assert np.allclose(zscored["image"].std(axis=0), [1, 1], atol=1e-3)
    
    # Test ZCA
    aligner.fit_zca(embs)
    whitened = aligner.transform_zca(embs)
    assert "text" in whitened and "image" in whitened
    # Verify shape isn't altered
    assert whitened["text"].shape == (2, 2)
