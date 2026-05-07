import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.models.forecaster import LagFeatureForecaster


def time_based_split(X, y, train_fraction=0.8):
    split_idx = int(len(X) * train_fraction)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def fit_scaler(X_tr):
    n, L, F = X_tr.shape
    scaler = StandardScaler()
    # By flattening each column will correspond to one specific (lag, feature) position
    # each (L,F) position being normalised by its own statistics ensures scale invariance for features
    X_tr_flat = X_tr.reshape(n, L * F)
    # scale only training data to prevent lookahead bias
    scaler.fit(X_tr_flat)
    return scaler


def apply_scaler(scaler, X):
    n, L, F = X.shape
    X_flat = X.reshape(n, L * F)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(n, L, F)


def make_dataloader(X, y, batch_size):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_training_loop(model, train_loader, val_loader, config):
    """Uses early stopping on validation mse loss to prevent overfitting."""
    learning_rate = config.get("learning_rate", 1e-3)
    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 10)
    grad_clip = config.get("grad_clip", 1.0)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    print(f"Training: max_epochs={max_epochs}, patience={patience}, lr={learning_rate}")

    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            y_hat, _ = model(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            # Gradient clipping prevents extreme market events causing infinite gradients
            # which destabilises training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_hat, _ = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if (epoch + 1) % 10 == 0:
            print(
                f"  epoch {epoch + 1:>3} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | no_improve={epochs_without_improvement}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"  early stopping at epoch {epoch + 1} | best_val_loss={best_val_loss:.6f}"
            )
            break

    model.load_state_dict(best_weights)
    print(f"best_val_loss={best_val_loss:.6f}")
    return model


class FittedAttentionModel:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.model.eval()

    def predict(self, X_test):
        X_scaled = apply_scaler(self.scaler, X_test)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_hat, _ = self.model(X_tensor)
        return y_hat.squeeze(1).numpy()

    def get_attention(self, X_test):
        X_scaled = apply_scaler(self.scaler, X_test)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            _, A = self.model(X_tensor)
        return A.numpy()


def train(X_train, y_train, config):
    """Trains a LagFeatureForecaster on a given training set and returns the model."""
    print(f"train() called | X_train={X_train.shape}, y_train={y_train.shape}")

    num_lags = X_train.shape[1]
    num_features = X_train.shape[2]

    X_tr, X_val, y_tr, y_val = time_based_split(X_train, y_train)

    scaler = fit_scaler(X_tr)
    X_tr_scaled = apply_scaler(scaler, X_tr)
    X_val_scaled = apply_scaler(scaler, X_val)

    batch_size = config.get("batch_size", 32)
    train_loader = make_dataloader(X_tr_scaled, y_tr, batch_size)
    val_loader = make_dataloader(X_val_scaled, y_val, batch_size)

    model = LagFeatureForecaster(
        num_lags=num_lags,
        num_features=num_features,
        embed_dim=config.get("embed_dim", 32),
        num_queries=config.get("num_queries", 4),
        head_config=config.get("head_config", None),
    )

    model = run_training_loop(model, train_loader, val_loader, config)

    return FittedAttentionModel(model=model, scaler=scaler)
