import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.models.forecaster import Forecaster
from src.models.lag_feature_forecaster import LagFeatureForecaster


SEED = 40304451
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GRAD_CLIP = 1.0


class AttentionForecaster(Forecaster):
    def __init__(self, config: dict):
        super().__init__(config)
        self.embed_dim = int(config.get("embed_dim", 32))
        self.num_queries = int(config.get("num_queries", 4))
        self.batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
        self.max_epochs = int(config.get("max_epochs", DEFAULT_MAX_EPOCHS))
        self.patience = int(config.get("patience", DEFAULT_PATIENCE))
        self.learning_rate = float(config.get("learning_rate", DEFAULT_LEARNING_RATE))
        self.grad_clip = float(config.get("grad_clip", DEFAULT_GRAD_CLIP))
        self.train_fraction = float(config.get("train_fraction", 0.8))

        self.model_ = None
        self.scaler_ = None
        self.fitted_ = False

    def _fit_scaler(self, X_train: np.ndarray) -> StandardScaler:
        n, L, F = X_train.shape
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(n, L * F))
        return scaler

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        n, L, F = X.shape
        X_scaled = self.scaler_.transform(X.reshape(n, L * F))
        return X_scaled.reshape(n, L, F)

    def _make_dataloader(self, X: np.ndarray, y: np.ndarray) -> DataLoader:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return DataLoader(
            TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=False
        )

    def _time_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        split_idx = int(len(X) * self.train_fraction)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def _run_training_loop(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> nn.Module:
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        print(
            f"  training | max_epochs={self.max_epochs}, patience={self.patience}, lr={self.learning_rate}"
        )

        for epoch in range(self.max_epochs):
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimiser.zero_grad()
                y_hat, _ = model(X_batch)
                loss = loss_fn(y_hat, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
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

            if epochs_without_improvement >= self.patience:
                print(
                    f"  early stopping at epoch {epoch + 1} | best_val_loss={best_val_loss:.6f}"
                )
                break

        model.load_state_dict(best_weights)
        print(f"  training complete | best_val_loss={best_val_loss:.6f}")
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "AttentionForecaster":

        n, L, F = X_train.shape

        X_tr, X_val, y_tr, y_val = self._time_split(X_train, y_train)

        self.scaler_ = self._fit_scaler(X_tr)
        X_tr_scaled = self._apply_scaler(X_tr)
        X_val_scaled = self._apply_scaler(X_val)

        train_loader = self._make_dataloader(X_tr_scaled, y_tr)
        val_loader = self._make_dataloader(X_val_scaled, y_val)

        torch.manual_seed(SEED)
        self.model_ = LagFeatureForecaster(
            num_lags=L,
            num_features=F,
            embed_dim=self.embed_dim,
            num_queries=self.num_queries,
            head_config=self.config.get("head_config", None),
        )

        self.model_ = self._run_training_loop(self.model_, train_loader, val_loader)
        self.fitted_ = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:

        X_scaled = self._apply_scaler(X_test)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            y_hat, _ = self.model_(X_tensor)

        y_pred = y_hat.squeeze(1).numpy()
        print(f"  y_pred mean={y_pred.mean():.6f}, std={y_pred.std():.6f}")
        return y_pred

    def get_attention(self, X_test: np.ndarray) -> np.ndarray:

        X_scaled = self._apply_scaler(X_test)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            _, A = self.model_(X_tensor)

        return A.numpy()
