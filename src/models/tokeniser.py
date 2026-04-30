import torch
import torch.nn as nn


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
        # register to device for GPU
        self.register_buffer("lag_indices", lag_indices)
        self.register_buffer("feature_indices", feature_indices)

        # Initial debugging
        print(
            f"  num_lags={num_lags}, num_features={num_features}, embed_dim={embed_dim}"
        )
        print(f"  total tokens per sample: {num_lags * num_features}")
        print(
            f"  scalar_embed params: {sum(p.numel() for p in self.scalar_embed.parameters())}"
        )

        print(
            f"  lag_embed params: {sum(p.numel() for p in self.lag_embed.parameters())}"
        )

        print(
            f"  feature_embed params: {sum(p.numel() for p in self.feature_embed.parameters())}"
        )

    def forward(self, x):
        # x shape = (batch, L, F)
        batch_size = x.shape[0]

        print(f"Tokeniser forward | input shape: {x.shape}")

        # scalar_embed expects its last dimension to be 1
        x_expanded = x.unsqueeze(-1)
        phi_out = self.scalar_embed(x_expanded)

        print(f"  phi_out shape: {phi_out.shape}")

        lag_vecs = self.lag_embed(self.lag_indices)
        feature_vecs = self.feature_embed(self.feature_indices)

        lag_vecs = lag_vecs.unsqueeze(1).expand(
            self.num_lags, self.num_features, self.embed_dim
        )
        feature_vecs = feature_vecs.unsqueeze(0).expand(
            self.num_lags, self.num_features, self.embed_dim
        )

        tokens = phi_out + lag_vecs + feature_vecs

        print(f"tokens shape before flatten: {tokens.shape}")

        tokens = tokens.reshape(
            batch_size,
            # collapse L and F into a single L x F sequence dimension
            self.num_lags * self.num_features,
            self.embed_dim,
        )

        print(f"tokens shape after flatten: {tokens.shape}")

        return tokens
