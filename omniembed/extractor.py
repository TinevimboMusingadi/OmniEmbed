import torch

class HiddenStateExtractor:
    """
    Context manager for extracting hidden states from multiple transformer layers
    during a forward pass using PyTorch hooks.
    """
    def __init__(self, model, layers=[-1, -2, -4], pool="mean"):
        """
        Args:
            model: The huggingface/torch model (e.g. Qwen3OmniMoeForConditionalGeneration)
            layers: List of layer indices to hook (Negative indices compute from the end)
            pool: Pooling method: "mean" or "eos"
        """
        self.model = model
        self.layers = layers
        self.pool = pool
        self.hidden_states_cache = {}
        self.hooks = []

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Transformers output is typically a tuple, where output[0] is the hidden state
            # shape: [batch, seq_len, hidden_dim]
            self.hidden_states_cache[layer_idx] = output[0].detach().cpu()
        return hook_fn

    def __enter__(self):
        # Clear cache from previous runs
        self.hidden_states_cache.clear()
        
        # Determine internal layers attribute (differs by model architecture)
        # For Qwen Models it's usually `model.model.layers`, adjust if needed
        model_layers = getattr(self.model, "model", self.model).layers
        
        for layer_idx in self.layers:
            actual_idx = layer_idx if layer_idx >= 0 else len(model_layers) + layer_idx
            hook = model_layers[actual_idx].register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_embedding(self, layer_idx, attention_mask=None):
        """
        Pools the stored hidden state into an embedding vector.
        
        Args:
            layer_idx: The hooked layer index.
            attention_mask: Tensor of shape [batch, seq_len] with 1 for active tokens.
        """
        if layer_idx not in self.hidden_states_cache:
            raise ValueError(f"No hidden state cached for layer {layer_idx}. Did the model run via the context manager?")
            
        last_hidden = self.hidden_states_cache[layer_idx] # on CPU
        
        if attention_mask is None:
            # Assume all tokens are valid if no mask provided
            attention_mask = torch.ones(last_hidden.shape[:2], dtype=torch.float32, device=last_hidden.device)
        else:
            attention_mask = attention_mask.cpu()
            
        mask_expanded = attention_mask.unsqueeze(-1).float()
        
        if self.pool == "mean":
            sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embedding = sum_hidden / count
        elif self.pool == "eos":
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            seq_lengths = torch.clamp(seq_lengths, min=0)
            embedding = last_hidden[torch.arange(last_hidden.size(0)), seq_lengths]
        else:
            raise ValueError(f"Unknown pooling mechanism: {self.pool}")
            
        return embedding
