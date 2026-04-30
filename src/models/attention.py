import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentQueryAttention(nn.Module):
    """
    We take U_t : R^{(L . F) x d}
    and produce
        Z_t : R^{m x d_v}
        plus attention weights:
        A_t : R^{m x (L . F)}
    """

    def __init__(self, embed_dim, num_queries, d_k=None, d_v=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.d_k = d_k if d_k is not None else embed_dim
        self.d_v = d_v if d_v is not None else embed_dim

        self.W_K = nn.Linear(embed_dim, self.d_k, bias=False)
        self.W_V = nn.Linear(embed_dim, self.d_v, bias=False)
        self.W_Q = nn.Linear(embed_dim, self.d_k, bias=False)

        self.Q_latent = nn.Parameter(torch.randn(num_queries, embed_dim))

        print(
            f"  embed_dim={embed_dim}, num_queries={num_queries}, d_k={self.d_k}, d_v={self.d_v}"
        )
        print(f"W_K params: {sum(p.numel() for p in self.W_K.parameters())}")
        print(f"W_V params: {sum(p.numel() for p in self.W_V.parameters())}")
        print(f"W_Q params: {sum(p.numel() for p in self.W_Q.parameters())}")
        print(f"Q_latent params: {self.Q_latent.numel()}")

    def forward(self, tokens):
        print(f"LatentQueryAttention forward | tokens shape: {tokens.shape}")

        K = self.W_K(tokens)
        V = self.W_V(tokens)

        Q = self.W_Q(self.Q_latent)

        print(f"K shape: {K.shape}")
        print(f"V shape: {V.shape}")
        print(f"Q shape: {Q.shape}")

        # Scaling keeps the dot products in a reasonable size to prevent saturating softmax
        scale = self.d_k**0.5
        # K has shape (batch, L*F, d_k)
        # Q has shape (m, d_k)
        # but with pytorch Q here is treated as (1, m, d_k) for the multiply
        # hence we transpose only the last two dimensions of K
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        print(f"  scores shape: {scores.shape}")

        A = F.softmax(scores, dim=-1)

        print(f"A shape: {A.shape}")
        print(f"A sum across tokens (should be 1.0): {A[0].sum(dim=-1).detach()}")

        Z = torch.matmul(A, V)

        print(f"Z shape: {Z.shape}")

        return Z, A
