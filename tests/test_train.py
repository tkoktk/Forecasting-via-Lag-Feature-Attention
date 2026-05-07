import numpy as np
from src.models.train import train

np.random.seed(40304451)

L = 5
F = 3
n_train = 200
n_test = 40

X_train = np.random.randn(n_train, L, F).astype(np.float32)
y_train = np.random.randn(n_train).astype(np.float32)
X_test = np.random.randn(n_test, L, F).astype(np.float32)

config = {
    "embed_dim": 16,
    "num_queries": 2,
    "batch_size": 32,
    "max_epochs": 30,
    "patience": 5,
    "learning_rate": 1e-3,
    "grad_clip": 1.0,
}

print("--- Test 1: training runs without error ---")
fitted = train(X_train, y_train, config)
print("Test 1 passed")

print("\n--- Test 2: predict returns correct shape ---")
y_pred = fitted.predict(X_test)
print(f"y_pred shape: {y_pred.shape}")
assert y_pred.shape == (n_test,), f"Shape mismatch: {y_pred.shape}"
print("Test 2 passed")

print("\n--- Test 3: predictions are finite ---")
assert np.all(np.isfinite(y_pred)), "Predictions contain nan or inf"
print(f"y_pred min: {y_pred.min():.6f}")
print(f"y_pred max: {y_pred.max():.6f}")
print(f"y_pred mean: {y_pred.mean():.6f}")
print(f"y_pred std: {y_pred.std():.6f}")
print("Test 3 passed")

print("\n--- Test 4: scaler fitted on train only ---")
scaler = fitted.scaler
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Expected shape: ({L * F},)")
assert scaler.mean_.shape == (L * F,), "Scaler shape mismatch"
print(f"Scaler mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"Scaler std range:  [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
print("Test 4 passed")

print("\n--- Test 5: get_attention returns correct shape ---")
A = fitted.get_attention(X_test)
num_queries = config["num_queries"]
print(f"A shape: {A.shape}")
assert A.shape == (n_test, num_queries, L * F), f"A shape mismatch: {A.shape}"
print(f"A min: {A.min():.6f}")
print(f"A max: {A.max():.6f}")
print(f"A row sums (should be 1.0): {A[0].sum(axis=-1)}")
print("Test 5 passed")

print("\n--- Test 6: mlp head override ---")
config_mlp = {**config, "head_config": {"type": "mlp", "hidden_dim": 32}}
fitted_mlp = train(X_train, y_train, config_mlp)
y_pred_mlp = fitted_mlp.predict(X_test)
assert y_pred_mlp.shape == (n_test,), f"Shape mismatch: {y_pred_mlp.shape}"
assert np.all(np.isfinite(y_pred_mlp)), "MLP predictions contain nan or inf"
print(f"MLP y_pred mean: {y_pred_mlp.mean():.6f}")
print(f"MLP y_pred std: {y_pred_mlp.std():.6f}")
print("Test 6 passed")
