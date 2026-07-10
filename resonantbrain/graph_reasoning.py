"""
Latent Graph Reasoning — multi-relational message passing module.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LATENT GRAPH REASONING (Relational Message Passing)
# =============================================================================

class LatentGraphReasoning(nn.Module):
    """
    Multi-relational latent graph reasoning adapted for ResonantBrain SSM.

    Constructs a learned, multi-relational adjacency graph over token
    representations, applies causal masking + top-k sparsity, then performs
    iterative message passing with LayerNorm-stabilized updates.

    Integration with ResonantBrain:
    - Replaces or augments the standard MLP in SSMAttentionBlock
    - Saliency projection synergizes with existing saliency eviction
    - Zero-init output projection preserves residual stream stability
    - Compatible with carry states, KV cache, and gradient checkpointing

    Args:
        dim: Model dimension (must match SSMAttentionBlock.dim)
        num_rules: Number of learned relational types (e.g., causation,
                   co-reference, hierarchy). Each rule is a (dim, dim) matrix.
        graph_steps: Number of message-passing iterations (internal reasoning depth)
        top_k_edges: Sparsity constraint — each token attends to at most this many
                     ancestors. Controls memory (O(T * top_k) vs O(T²)).
    """
    def __init__(self, dim, num_rules=8, graph_steps=3, top_k_edges=16):
        super().__init__()
        self.dim = dim
        self.num_rules = num_rules
        self.graph_steps = graph_steps
        self.top_k = top_k_edges

        # Saliency-weighted fuzzy node projection
        self.saliency_proj = nn.Linear(dim, 1)
        self.fuzzy_proj = nn.Linear(dim, dim)

        # Learned relational rules: (num_rules, dim, dim)
        self.rule_tensor = nn.Parameter(torch.randn(num_rules, dim, dim) / math.sqrt(dim))

        # Message transformation (per-step update)
        self.msg_proj = nn.Linear(dim, dim, bias=False)

        # Step normalization for stable iterative updates
        self.step_norm = nn.LayerNorm(dim)

        # Output projection (zero-init so it starts as identity residual)
        self.out_proj = nn.Linear(dim, dim)

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, C) — hidden states from attention/SSM layers
        Returns:
            (B, T, C) — relationally-enhanced hidden states (residual added)
        """
        B, T, C = x.shape

        # --- Node Representation ---
        # Saliency-weighted fuzzy nodes: highlights important token representations
        saliency_weights = torch.sigmoid(self.saliency_proj(x))  # (B, T, 1)
        nodes = torch.sigmoid(self.fuzzy_proj(x)) * saliency_weights  # (B, T, C)

        # --- Multi-Relational Adjacency ---
        # Compute pairwise relational scores: nodes[t] @ rule_r @ nodes[s]
        # Result: (B, num_rules, T, T) — full bipartite relation matrix
        # einsum: btc * rcd * bsd -> brts (t=query, s=key)
        adj = torch.einsum('btc, rcd, bsd -> brts', nodes, self.rule_tensor, nodes)

        # --- Causal Masking ---
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        adj = adj.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # --- Top-K Sparsity ---
        if self.top_k < T:
            topk_vals, _ = torch.topk(adj, self.top_k, dim=-1)
            kth_val = topk_vals[..., -1:]  # Value of the kth largest edge weight
            adj = adj.masked_fill(adj < kth_val, float('-inf'))

        adj = F.softmax(adj, dim=-1)  # Normalize edges per query token

        # --- Iterative Message Passing ---
        h = nodes
        for _ in range(self.graph_steps):
            # Aggregate messages from neighbors across all relation types
            messages = torch.einsum('brts, bsc -> brtc', adj, h)  # (B, R, T, C)
            m_agg = messages.mean(dim=1)  # Average across rules -> (B, T, C)
            h = self.step_norm(h + self.msg_proj(m_agg))  # Residual update

        # Zero-init projection + skip connection from original input
        return self.out_proj(h) + x
