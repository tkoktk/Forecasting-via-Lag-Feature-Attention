import torch
from src.models.forecaster import LagFeatureForecaster

L = 5
F = 3
d = 8
m = 4
batch_size = 4

print("--- Test 1: default head (linear) ---")
model_linear = LagFeatureForecaster(
    num_lags=L, num_features=F, embed_dim=d, num_queries=m
)

x = torch.randn(batch_size, L, F)
y_hat, A = model_linear(x)

assert y_hat.shape == (batch_size, 1), f"y_hat shape mismatch: {y_hat.shape}"
assert A.shape == (batch_size, m, L * F), f"A shape mismatch: {A.shape}"
print(f"y_hat shape: {y_hat.shape}")
print(f"A shape: {A.shape}")
print("Test 1 passed")

print("\n---Test 2: explicit linear head ---")
model_explicit = LagFeatureForecaster(
    num_lags=L,
    num_features=F,
    embed_dim=d,
    num_queries=m,
    head_config={"type": "linear"},
)
y_hat_2, _ = model_explicit(x)
assert y_hat_2.shape == (batch_size, 1), f"y_hat shape mismatch: {y_hat_2.shape}"
print("Test 2 passed")

print("\n--- Test 3: mlp head ---")
model_mlp = LagFeatureForecaster(
    num_lags=L,
    num_features=F,
    embed_dim=d,
    num_queries=m,
    head_config={"type": "mlp", "hidden_dim": 32},
)
y_hat_3, _ = model_mlp(x)
assert y_hat_3.shape == (batch_size, 1), f"y_hat shape mismatch: {y_hat_3.shape}"
print("Test 3 passed")

print("\n--- Test 4: bad head type raises error ---")
try:
    model_bad = LagFeatureForecaster(
        num_lags=L,
        num_features=F,
        embed_dim=d,
        num_queries=m,
        head_config={"type": "transformer"},
    )
    print("Test 4 FAILED - should have raised ValueError")
except ValueError as e:
    print(f"Caught expected error: {e}")
    print("Test 4 passed")

print("\n--- Test 5: attention weight sanity ---")
print(f"A sum across tokens (batch=0, query=0): {A[0, 0].sum().item():.6f}")
print(f"A sum across tokens (batch=0, query=1): {A[0, 1].sum().item():.6f}")
print(f"A min: {A.min().item():.6f}")
print(f"A max: {A.max().item():.6f}")

print("\n--- Test 6: feature and lag weight decomposition ---")
A_detached = A.detach()

A_lf = A_detached.reshape(batch_size, m, L, F)
print(f"A reshaped to (batch, m, L, F): {A_lf.shape}")

feature_weights = A_lf.sum(dim=1).sum(dim=1)
print(f"Feature weights shape (batch, F): {feature_weights.shape}")
print(f"Feature weights batch=0: {feature_weights[0]}")

lag_weights = A_lf.sum(dim=1).sum(dim=2)
print(f"Lag weights shape (batch, L): {lag_weights.shape}")
print(f"Lag weights batch=0: {lag_weights[0]}")

print(
    f"Feature weights sum (should equal m={m}): {feature_weights[0].sum().item():.4f}"
)
print(f"Lag weights sum (should equal m={m}): {lag_weights[0].sum().item():.4f}")
