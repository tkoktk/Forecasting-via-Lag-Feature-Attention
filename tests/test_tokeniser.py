import torch
from src.models.tokeniser import LagFeatureTokeniser

L = 5
F = 3
d = 8
batch_size = 4

tokeniser = LagFeatureTokeniser(num_lags=L, num_features=F, embed_dim=d)

x = torch.randn(batch_size, L, F)
print(f"\nInput tensor shape: {x.shape}")

tokens = tokeniser(x)

print(f"\nExpected output shape: ({batch_size}, {L * F}, {d})")
print(f"Actual output shape: {tokens.shape}")

assert tokens.shape == (batch_size, L * F, d), "Shape mismatch"
print("\nShape assertion passed")

print(f"\nToken value sample (batch=0, token=0): {tokens[0, 0, :].detach()}")
print(f"Token value sample (batch=0, token=1): {tokens[0, 1, :].detach()}")
print(f"Token value sample (batch=1, token=0): {tokens[1, 0, :].detach()}")

print(
    f"\nAre batch 0 and batch 1 token 0 different: {not torch.allclose(tokens[0, 0], tokens[1, 0])}"
)
print(
    f"Are token 0 and token 1 in batch 0 different: {not torch.allclose(tokens[0, 0], tokens[0, 1])}"
)
