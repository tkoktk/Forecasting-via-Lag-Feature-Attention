import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.models.forecaster import Forecaster


SEED = 40304451
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_CV_SPLITS = 3


# Overrides our base Forecaster
class RidgeForecaster(Forecaster):
    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha_grid = tuple(config.get("alpha_grid", DEFAULT_ALPHA_GRID))
        self.cv_splits = int(config.get("cv_splits", DEFAULT_CV_SPLITS))

        self.scaler_ = None
        self.model_ = None
        self.best_alpha_ = None
        self.fitted_ = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RidgeForecaster":
        """Train with Ridge Model"""
        n = X_train.shape[0]

        if n < self.min_train_points:
            # Some defence if our sample size is too small
            print(
                f"---------WARNING: insufficient training data: {n} rows (min={self.min_train_points}), fitting mean fallback -------------"
            )
            self.best_alpha_ = None
            self.fitted_ = False
            self._fallback_mean_ = float(np.nanmean(y_train))
            return self

        X_flat = X_train.reshape(n, -1)

        self.scaler_ = StandardScaler()
        # Only scale on training data
        X_scaled = self.scaler_.fit_transform(X_flat)

        n_splits = min(self.cv_splits, max(2, n // 200))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_alpha, best_mse = None, np.inf

        for alpha in self.alpha_grid:
            fold_mses = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_tr, X_va = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_va = y_train[train_idx], y_train[val_idx]
                # Fit a Ridge model
                model = Ridge(alpha=alpha, random_state=SEED)
                model.fit(X_tr, y_tr)
                fold_mses.append(mean_squared_error(y_va, model.predict(X_va)))

            avg_mse = float(np.mean(fold_mses)) if fold_mses else np.inf
            # Important for finetuning our regularisation alpha
            print(f"----alpha={alpha} | avg_mse={avg_mse:.6f}")

            if avg_mse < best_mse:
                best_mse, best_alpha = avg_mse, alpha

        self.best_alpha_ = float(best_alpha)
        print(f"best_alpha={self.best_alpha_} | best_mse={best_mse:.6f}")

        self.model_ = Ridge(alpha=self.best_alpha_, random_state=SEED)
        self.model_.fit(X_scaled, y_train)
        self.fitted_ = True

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predictions of next-H-day cumulative log returns."""

        n = X_test.shape[0]

        if not self.fitted_:
            print(f"fallback mean prediction: {self._fallback_mean_:.6f}")
            return np.full(n, self._fallback_mean_)

        X_flat = X_test.reshape(n, -1)
        X_scaled = self.scaler_.transform(X_flat)
        y_pred = self.model_.predict(X_scaled)

        print(f"y_pred mean={y_pred.mean():.6f}, std={y_pred.std():.6f}")
        return y_pred
