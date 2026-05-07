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

    def forward(self, tokens):

        K = self.W_K(tokens)
        V = self.W_V(tokens)

        Q = self.W_Q(self.Q_latent)

        # Scaling keeps the dot products in a reasonable size to prevent saturating softmax
        scale = self.d_k**0.5
        # K has shape (batch, L*F, d_k)
        # Q has shape (m, d_k)
        # but with pytorch Q here is treated as (1, m, d_k) for the multiply
        # hence we transpose only the last two dimensions of K
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Attention: softmax of ((Q . K^T) / sqr(d_k)
        A = F.softmax(scores, dim=-1)
        Z = torch.matmul(A, V)

        return Z, A


class LagFeatureTokeniser(nn.Module):
    """
    Our tokeniser takes a single lookback window
        X : R^{L * F}
    and produces the token matrix:
        U_t : R^{(L . F) x d}
    """

    def __init__(self, num_lags, num_features, embed_dim):
        super().__init__()

        self.num_lags = num_lags
        self.num_features = num_features
        self.embed_dim = embed_dim

        # Attention operates on vectors, but our cumulative
        # log returns are scalars.
        # We can lift each scalar into an embedding space
        # With this linear transformation
        self.scalar_embed = nn.Linear(1, embed_dim)
        # We shall also add the lag embedding and feature embedding:
        self.lag_embed = nn.Embedding(num_lags, embed_dim)
        self.feature_embed = nn.Embedding(num_features, embed_dim)

        lag_indices = torch.arange(num_lags)
        feature_indices = torch.arange(num_features)
        # register to device for potential GPU use
        self.register_buffer("lag_indices", lag_indices)
        self.register_buffer("feature_indices", feature_indices)

    def forward(self, x):
        # x shape = (batch, L, F)
        batch_size = x.shape[0]

        # scalar_embed expects its last dimension to be 1
        x_expanded = x.unsqueeze(-1)

        phi_out = self.scalar_embed(x_expanded)  # phi(x_{t-l,f})

        lag_vecs = self.lag_embed(self.lag_indices)  # p_l
        feature_vecs = self.feature_embed(self.feature_indices)  # e_f

        lag_vecs = lag_vecs.unsqueeze(1).expand(
            self.num_lags, self.num_features, self.embed_dim
        )
        feature_vecs = feature_vecs.unsqueeze(0).expand(
            self.num_lags, self.num_features, self.embed_dim
        )

        # So we have u_{t,l,f} = phi(x_{t-l,f}) + p_l + e_f
        tokens = phi_out + lag_vecs + feature_vecs

        tokens = tokens.reshape(
            batch_size,
            # collapse L and F into a single L x F sequence dimension
            self.num_lags * self.num_features,
            self.embed_dim,
        )

        return tokens
