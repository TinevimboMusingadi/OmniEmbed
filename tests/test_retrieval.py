import numpy as np
from omniembed.retrieval import calculate_recall_at_k, compute_modality_gap

def test_calculate_recall_at_k():
    # Identity matrix means each item only matches perfectly with itself
    q = np.array([[1, 0], [0, 1]])
    t = np.array([[1, 0], [0, 1]])
    
    # Expect 100% since index i matches index i 
    recalls = calculate_recall_at_k(q, t, k_list=[1, 5])
    assert recalls["Recall@1"] == 1.0
    
    # Reverse so items perfectly miss their expected target pairing
    t_bad = np.array([[0, 1], [1, 0]])
    recalls_bad = calculate_recall_at_k(q, t_bad, k_list=[1])
    assert recalls_bad["Recall@1"] == 0.0

def test_compute_modality_gap():
    a = np.array([[0, 1.], [0, 2.]])  # Mean: [0, 1.5]
    b = np.array([[3, 1.], [3, 2.]])  # Mean: [3, 1.5]
    
    gap = compute_modality_gap(a, b)
    
    # L2 dist between [0, 1.5] and [3, 1.5] is 3
    assert np.isclose(gap["l2_gap"], 3.0)
    
    # Because they are strictly parallel on the specific projection, 
    # angle shouldn't be zero meaning the gap is > 0
    assert gap["cosine_gap"] > 0.0
