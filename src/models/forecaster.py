import torch.nn as nn
from src.models.tokeniser import LagFeatureTokeniser
from src.models.attention import LatentQueryAttention


def build_head(input_dim, head_config=None):
    print(f"Building head | input_dim={input_dim}")

    if head_config is None or head_config.get("type") == "linear":
        return nn.Linear(input_dim, 1)

    if head_config.get("type") == "mlp":
        hidden_dim = head_config.get("hidden_dim", 32)
        print(f"Using mlp head with hidden_dim={hidden_dim}")
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    raise ValueError(f"Unknown head type: {head_config.get('type')}")


class LagFeatureForecaster(nn.Module):
    """
    Wires the tokeniser.py and attention.py together
    Then applies a small MLP head to produce
        ŷ_{t+h}

    """

    def __init__(
        self, num_lags, num_features, embed_dim=32, num_queries=4, head_config=None
    ):
        super().__init__()

        if head_config is None:
            head_config = {"type": "mlp", "hidden_dim": 32}

        self.num_lags = num_lags
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.tokeniser = LagFeatureTokeniser(
            num_lags=num_lags, num_features=num_features, embed_dim=embed_dim
        )

        self.attention = LatentQueryAttention(
            embed_dim=embed_dim, num_queries=num_queries
        )

        head_input_dim = num_queries * embed_dim
        self.head = build_head(head_input_dim, head_config)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"LagFeatureForecaster init")
        print(f"  num_lags={num_lags}, num_features={num_features}")
        print(f"  embed_dim={embed_dim}, num_queries={num_queries}")
        print(f"  head_config={head_config}")
        print(f"  head_input_dim={head_input_dim}")
        print(f"  total trainable params: {total_params}")

    def forward(self, x):
        print(f"Forecaster forward | input shape: {x.shape}")

        tokens = self.tokeniser(x)
        Z, A = self.attention(tokens)

        print(f"  Z shape: {Z.shape}")

        z_flat = Z.reshape(x.shape[0], -1)

        print(f"  z_flat shape: {z_flat.shape}")

        y_hat = self.head(z_flat)

        print(f"  y_hat shape: {y_hat.shape}")

        return y_hat, A
