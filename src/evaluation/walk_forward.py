import pandas as pd

from src.data.features import forward_log_return
from src.data.loader import get_ticker_data
from src.evaluation.metrics import compute_metrics
from src.models.ridge_baseline import Forecaster


def walk_forward(prices, config, horizon=5, step=10):
    """Evaluate a Forecaster configuration via expanding-window walk-forward.

    Trains a fresh Forecaster at the start of each block of ``step`` test dates,
    using all data strictly before the block's first date, then predicts the
    full block with that single model.

    The training window expands forward over time; the test set never overlaps with
    training data.

    Args:
        prices: OHLC DataFrame with a DatetimeIndex and at least a ``Close`` column.
            Note: Higher-frequency columns (``High``, ``Low``) are required if the config uses Parkinson volatility.

        config: Dictionary passed to the Forecaster constructor that specifies features and model hyperparams.

        horizon: Forecast horizon in periods. The target is the cumulative log return from t to t+horizon.

        step: Number of test dates to predict between retrains. Larger values are cheaper but use staler models within each block.

    Returns:
        A tuple ``(y_true, y_pred)`` of aligned pandas Series indexed by test date.
        Dates where the target is undefined (the final ``horizon`` rows) are excluded.


        Predictions within a block are made by a single model trained before the block began; this is a deliberate compute tradeoff, not a bug.
        Set ``step=1`` for daily-refit evaluation.
    """

    y_full = forward_log_return(prices["Close"], horizon=horizon)
    y_full.name = "y_true"
    test_dates = y_full.dropna().index

    all_preds = []

    for i in range(0, len(test_dates), step):
        block = test_dates[i : i + step]
        first_test = block[0]

        X_train = prices.loc[: first_test - pd.Timedelta(days=1)]
        y_train = y_full.loc[: first_test - pd.Timedelta(days=1)]

        model = Forecaster(config=config)
        model.fit(X_train, y_train)

        X_pred = prices.loc[: block[-1]]
        y_hat = model.predict(X_pred)

        all_preds.append(y_hat.reindex(block).dropna())

    y_pred = (
        pd.concat(all_preds) if all_preds else pd.Series(dtype=float, name="y_pred")
    )
    y_true = y_full.reindex(y_pred.index)

    return y_true, y_pred


def run_evaluation(
    configs,
    prices_df,
    tickers,
    date_start,
    date_end,
    horizon,
    step,
    verbose=True,
    return_predictions=False,
):
    """Evaluate multiple forecasting configurations across tickers using walk-forward validation.
    For each configuration and ticker combination, performs walk-forward validatio
    and computes directional accuracy, MAE, and RMSE metrics.
    """
    results = []

    for config_name, config_params in configs.items():
        if verbose:
            print(f"\n{'_' * 60}")
            print(f"Config: {config_name}")
            print(f"Params: {config_params}\n")

        for ticker in tickers:
            ticker_prices = get_ticker_data(prices_df, ticker)
            ticker_prices = ticker_prices.loc[date_start:date_end]

            y_true, y_pred = walk_forward(
                ticker_prices, config_params, horizon=horizon, step=step
            )
            metrics = compute_metrics(y_true, y_pred)

            if return_predictions:
                for date, true_val, pred_val in zip(
                    y_true.index, y_true.values, y_pred.values
                ):
                    results.append(
                        {
                            "config": config_name,
                            "ticker": ticker,
                            "date": date,
                            "y_true": true_val,
                            "y_pred": pred_val,
                            "ic": metrics["ic"],
                            "p_value": metrics["p_value"],
                            "diracc": metrics["diracc"],
                            "mae": metrics["mae"],
                            "rmse": metrics["rmse"],
                        }
                    )
            else:
                results.append(
                    {
                        "config": config_name,
                        "ticker": ticker,
                        "ic": metrics["ic"],
                        "p_value": metrics["p_value"],
                        "diracc": metrics["diracc"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                    }
                )

            if verbose:
                print(
                    f"{ticker}: IC={metrics['ic']:.4f} (p={metrics['p_value']:.4f}), "
                    f"DirAcc={metrics['diracc']:.4f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}"
                )

    results_df = pd.DataFrame(results)

    if verbose:
        print("\nMean results by config:")
        print(
            results_df.groupby("config")[["ic", "diracc", "mae", "rmse"]]
            .mean()
            .sort_values("ic", ascending=False)
        )

    return results_df
