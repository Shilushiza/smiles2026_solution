from __future__ import annotations
import torch

def aggregate(hidden_states, attention_mask, response_start=None):
    dev = hidden_states.device
    real_mask = attention_mask.bool().to(dev)
    n_real = int(real_mask.sum().item())
    n_layers = hidden_states.shape[0]
    use_response = response_start is not None and response_start < n_real - 1

    def _mean_pool(layer_tensor):
        if use_response:
            tokens = layer_tensor[response_start:n_real]
        else:
            tokens = layer_tensor[real_mask]
        return tokens.mean(dim=0)

    feats = [_mean_pool(hidden_states[i]) for i in range(1, n_layers)]
    return torch.cat(feats)  # 24*896 = 21504

def extract_geometric_features(hidden_states, attention_mask):
    return torch.zeros(0)

def aggregation_and_feature_extraction(hidden_states, attention_mask, use_geometric=False, response_start=None):
    agg = aggregate(hidden_states, attention_mask, response_start=response_start)
    if use_geometric:
        return torch.cat([agg, extract_geometric_features(hidden_states, attention_mask)])
    return agg
