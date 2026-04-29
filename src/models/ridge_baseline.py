import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import log_returns, parkinson_volatility


SEED = 40304451


class Forecaster:
    def __init__(self, config=None, random_state=SEED):
        # Set defaults:
        # 5 lags was adopted in previous study
        self.n_lags = 5
        self.parkinson_vol_windows = ()
        self.model_type = "ridge"
        self.alpha_grid = (0.01, 0.1, 1.0, 10.0)
        self.cv_splits = 3
        self.min_train_points = 200
        self.random_state = random_state

        # Override default feature values with config details
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        self.n_lags = int(self.n_lags) if self.n_lags else 0
        self.parkinson_vol_windows = (
            tuple(int(w) for w in self.parkinson_vol_windows)
            if self.parkinson_vol_windows
            else ()
        )
        self.model_type = str(self.model_type).lower()
        self.alpha_grid = tuple(float(a) for a in self.alpha_grid)
        self.cv_splits = int(self.cv_splits)
        self.min_train_points = int(self.min_train_points)
        self.random_state = int(self.random_state)

        self.pipe_ = None
        self.best_alpha_ = None
        self.fitted_ = False

    @staticmethod
    def _finite_mean(y):
        y = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(y) == 0:
            return 0.0
        mean = float(y.mean())
        return mean if np.isfinite(mean) else 0.0

    def _make_features(self, X):
        """
        Constructs a feature matrix of lagged log returns and (optionally) Parkinson volatility features from raw OHLC Data.

        To prevent look-ahead bias in our walk forward, features are shifted to use information available strictly before time t.

        Args:
            X: OHLC DataFrame with at least a ``Close`` column.
            ``High`` and ``Low`` are required if ``parkinson_vol_windows`` is set.
            Indexed by date.

        Returns:
            Feature matrix with one column per feature and a DatetimeIndex aligned to ``X``.
            Rows containing any NaN are dropped, so the returned index is a strict subset of ``X.index``.

        Notes:
            Returns an empty DataFrame (with X's index) when no features are configured. This keeps the downstream ``fit`` logic uniform: it
            falls back to a DummyRegressor rather than raising.
        """
        close = X["Close"].astype(float)
        lr = log_returns(close)

        feats = {}

        # Make Raw Lag features
        if self.n_lags > 0:
            for i in range(1, self.n_lags + 1):
                feats[f"lag_{i}"] = lr.shift(i)

        if self.parkinson_vol_windows:
            # Parkinson Volatility needs OHLC Data
            if all(col in X.columns for col in ["High", "Low"]):
                for w in self.parkinson_vol_windows:
                    feats[f"parkinson_vol_{w}"] = parkinson_volatility(
                        X["High"], X["Low"], w
                    ).shift(1)

        if not feats:
            print("WARNING: No features constructed, returning empty df")
            return pd.DataFrame(index=X.index)

        feature_df = (
            pd.DataFrame(feats, index=X.index)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        print(
            f"Features constructed: {feature_df.shape[1]} columns, {len(feature_df)} rows after dropna"
        )
        return feature_df

    def _get_model(self, alpha=None):
        return Ridge(alpha=alpha or 1.0, random_state=self.random_state)

    def _fallback_dummy(self, y):
        mean_y = self._finite_mean(y)
        pipe = Pipeline(
            [("model", DummyRegressor(strategy="constant", constant=mean_y))]
        )
        pipe.fit([[0.0]], [0.0])
        self.pipe_ = pipe
        self.best_alpha_ = None
        self.fitted_ = True
        print(f"Fallback dummy fitted with constant={mean_y:.6f}")
        return self

    def fit(self, X_train, y_train):
        feature_df = self._make_features(X_train)

        if feature_df.empty:
            return self._fallback_dummy(y_train)

        y = y_train.reindex(feature_df.index)
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        feature_df = feature_df.loc[mask]
        y = y.loc[mask]

        if len(y) == 0 or len(feature_df) < self.min_train_points:
            print(
                f"Insufficient training data: {len(feature_df)} rows (min={self.min_train_points})"
            )
            return self._fallback_dummy(y_train)

        # Tune alpha with scaling
        n_splits = min(self.cv_splits, max(2, len(feature_df) // 200))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_alpha, best_mse = None, np.inf
        for alpha in self.alpha_grid:
            fold_mses = []
            for train_idx, val_idx in tscv.split(feature_df.values):
                X_tr, X_va = feature_df.values[train_idx], feature_df.values[val_idx]
                y_tr, y_va = y.values[train_idx], y.values[val_idx]

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", self._get_model(alpha=alpha)),
                    ]
                )
                pipe.fit(X_tr, y_tr)
                fold_mses.append(mean_squared_error(y_va, pipe.predict(X_va)))

            avg_mse = float(np.mean(fold_mses)) if fold_mses else np.inf
            if avg_mse < best_mse:
                best_mse, best_alpha = avg_mse, alpha

        self.best_alpha_ = float(best_alpha or 1.0)
        print(f"Best alpha: {self.best_alpha_} (CV MSE={best_mse:.6f})")

        self.pipe_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", self._get_model(alpha=self.best_alpha_)),
            ]
        )
        self.pipe_.fit(feature_df.values, y.values)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Returns predictions of next-H-day cumulative log returns.
        """
        feature_df = self._make_features(X)

        if not self.fitted_ or self.pipe_ is None or feature_df.empty:
            print("WARNING: Model not fitted or no features. Returning zeros")
            return pd.Series(0.0, index=X.index, name="y_pred")

        y_hat = self.pipe_.predict(feature_df.values)
        return pd.Series(y_hat, index=feature_df.index, name="y_pred")
