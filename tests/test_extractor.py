import torch
import torch.nn as nn
from omniembed.extractor import HiddenStateExtractor

class MockLayer(nn.Module):
    def forward(self, x):
        # Emulate transformer layer returning tuple
        return (x * 2,)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(4)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x

class MockWrapper:
    def __init__(self):
        self.model = MockModel()

def test_extractor_hooks_and_pooling():
    model = MockWrapper()
    extractor = HiddenStateExtractor(model, layers=[-1, -2], pool="mean")
    
    x = torch.ones(2, 5, 10)  # batch_size=2, seq_len=5, hidden_dim=10
    
    with extractor:
        model.model(x)
        
    emb1 = extractor.get_embedding(-1)
    emb2 = extractor.get_embedding(-2)
    
    assert emb1.shape == (2, 10)
    assert emb2.shape == (2, 10)
    
    # Test values (since each layer multiplies by 2, Layer -1 is 2^4 = 16)
    # Layer -2 is 2^3 = 8
    assert torch.allclose(emb1, torch.ones(2, 10) * 16)
    assert torch.allclose(emb2, torch.ones(2, 10) * 8)
