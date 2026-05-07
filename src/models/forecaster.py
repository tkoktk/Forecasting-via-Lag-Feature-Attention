import numpy as np
import pandas as pd

from src.data.features import log_returns, forward_log_return, parkinson_volatility


class Forecaster:
    """Parent class - carries main forecaster functionality including building features from a config dict"""

    def __init__(self, config: dict):
        self.n_lags = int(config.get("n_lags", 5))
        self.parkinson_vol_windows = tuple(config.get("parkinson_vol_windows", ()))
        self.min_train_points = int(config.get("min_train_points", 200))
        self.config = config

        print()
        print(
            f"\n{self.__class__.__name__} | n_lags={self.n_lags}, pvol_windows={self.parkinson_vol_windows}"
        )

    def _make_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a feature matrix (for example of lagged log returns) from raw OHLC Data.
        Feature construction is dependent on the config passed in.

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

        close = prices["Close"].astype(float)
        ret = log_returns(close)

        feats = {}

        pvol_series = {}
        if self.parkinson_vol_windows:
            missing = [c for c in ["High", "Low"] if c not in prices.columns]
            if missing:
                raise ValueError(
                    f"_make_features: columns {missing} required for Parkinson vol but not found"
                )

            for w in self.parkinson_vol_windows:
                pvol_series[w] = parkinson_volatility(prices["High"], prices["Low"], w)

        for lag in range(1, self.n_lags + 1):
            feats[f"ret_lag{lag}"] = ret.shift(lag)
            for w, pvol in pvol_series.items():
                feats[f"pvol_{w}_lag{lag}"] = pvol.shift(lag)

        if not feats:
            print("WARNING: no features configured, returning empty DataFrame")
            return pd.DataFrame(index=prices.index)

        feature_df = (
            pd.DataFrame(feats, index=prices.index)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        print(
            f"_make_features | shape={feature_df.shape}, columns={list(feature_df.columns)}"
        )
        return feature_df

    def build_features(
        self, prices: pd.DataFrame, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Constructs (X, y, dates) from raw OHLCV prices.

        X has shape (N, L, F) where:
            N = number of valid samples
            L = n_lags
            F = number of features (1 for lags only, 2+ with volatility)

        Args:
            prices: DataFrame with at least a Close column.
            horizon: Forecast horizon h. Target is the h-day forward log return.

        Returns:
            X:     np.ndarray of shape (N, L, F)
            y:     np.ndarray of shape (N,)
            dates: pd.DatetimeIndex of length N
        """
        print(f"build_features | horizon={horizon}")

        feature_df = self._make_features(prices)
        target = forward_log_return(prices["Close"].astype(float), horizon).rename(
            "target"
        )

        combined = pd.concat([feature_df, target], axis=1).dropna()

        n_features = 1 + len(self.parkinson_vol_windows)
        X = combined.drop(columns="target").values.reshape(
            len(combined), self.n_lags, n_features
        )
        y = combined["target"].values
        dates = combined.index

        print(f"build_features | X={X.shape}, y={y.shape}")
        print(f"  date range: {dates[0].date()} to {dates[-1].date()}")
        print(f"  y mean={y.mean():.6f}, std={y.std():.6f}")

        return X, y, dates

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Forecaster":
        raise NotImplementedError(f"{self.__class__.__name__} must implement fit()")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement predict()")


# class Forecaster:
#     """Parent class - carries main forecaster functionality including building features from a config dict"""

#     def __init__(self, config: dict):
#         self.n_lags = int(config.get("n_lags", 5))
#         self.parkinson_vol_windows = tuple(config.get("parkinson_vol_windows", ()))
#         self.min_train_points = int(config.get("min_train_points", 200))
#         self.config = config

#     def _make_features(self, prices: pd.DataFrame) -> pd.DataFrame:
#         """
#         Constructs a feature matrix (for example of lagged log returns) from raw OHLC Data.
#         Feature construction is dependent on the config passed in.

#         To prevent look-ahead bias in our walk forward, features are shifted to use information available strictly before time t.

#         Args:
#             X: OHLC DataFrame with at least a ``Close`` column.
#             ``High`` and ``Low`` are required if ``parkinson_vol_windows`` is set.
#             Indexed by date.

#         Returns:
#             Feature matrix with one column per feature and a DatetimeIndex aligned to ``X``.
#             Rows containing any NaN are dropped, so the returned index is a strict subset of ``X.index``.

#         Notes:
#             Returns an empty DataFrame (with X's index) when no features are configured. This keeps the downstream ``fit`` logic uniform: it
#             falls back to a DummyRegressor rather than raising.
#         """

#         close = prices["Close"].astype(float)
#         ret = log_returns(close)

#         feats = {}

#         for lag in range(1, self.n_lags + 1):
#             feats[f"ret_lag{lag}"] = ret.shift(lag)

#         if self.parkinson_vol_windows:
#             missing = [c for c in ["High", "Low"] if c not in prices.columns]
#             if missing:
#                 raise ValueError(
#                     f"_make_features: columns {missing} required for Parkinson vol but not found"
#                 )

#             for w in self.parkinson_vol_windows:
#                 feats[f"pvol_{w}"] = parkinson_volatility(
#                     prices["High"], prices["Low"], w
#                 ).shift(1)

#         if not feats:
#             print("WARNING: no features in config, returning empty DataFrame")
#             return pd.DataFrame(index=prices.index)

#         feature_df = (
#             pd.DataFrame(feats, index=prices.index)
#             .replace([np.inf, -np.inf], np.nan)
#             .dropna()
#         )

#         return feature_df

#     def build_features(
#         self, prices: pd.DataFrame, horizon: int
#     ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
#         """
#         Constructs (X, y, dates) from raw OHLCV prices.

#         X has shape (N, L, F) where:
#             N = number of valid samples
#             L = n_lags
#             F = number of features (1 for lags only, 2+ with volatility)

#         Args:
#             prices: DataFrame with at least a Close column.
#             horizon: Forecast horizon h. Target is the h-day forward log return.

#         Returns:
#             X:     np.ndarray of shape (N, L, F)
#             y:     np.ndarray of shape (N,)
#             dates: pd.DatetimeIndex of length N
#         """

#         feature_df = self._make_features(prices)
#         target = forward_log_return(prices["Close"].astype(float), horizon).rename(
#             "target"
#         )

#         combined = pd.concat([feature_df, target], axis=1).dropna()

#         n_features = 1 + len(self.parkinson_vol_windows)
#         X = combined.drop(columns="target").values.reshape(
#             len(combined), self.n_lags, n_features
#         )
#         y = combined["target"].values
#         dates = combined.index

#         return X, y, dates

#     # This is a base class only. Each subclass will override fit() and predict() according to the model
#     def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Forecaster":
#         raise NotImplementedError(f"{self.__class__.__name__} must implement fit()")

#     def predict(self, X_test: np.ndarray) -> np.ndarray:
#         raise NotImplementedError(f"{self.__class__.__name__} must implement predict()")
