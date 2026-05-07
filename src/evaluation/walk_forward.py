import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_metrics


def walk_forward(
    X, y, dates, forecaster_cls, config, eval_start, eval_end, horizon, step
):
    dates = pd.DatetimeIndex(dates)
    eval_mask = (dates >= eval_start) & (dates <= eval_end)
    eval_dates = dates[eval_mask]

    if len(eval_dates) == 0:
        raise ValueError(f"No samples found between {eval_start} and {eval_end}")

    fold_results = []

    for i in range(0, len(eval_dates), step):
        block_dates = eval_dates[i : i + step]
        first_test_date = block_dates[0]

        train_mask = dates < first_test_date
        X_train = X[train_mask]
        y_train = y[train_mask]

        test_mask = np.isin(dates, block_dates)
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_dates_block = dates[test_mask]

        print(
            f"fold {i // step + 1} | train={len(X_train)}, test={len(X_test)}, first_test={first_test_date.date()}"
        )

        if len(X_train) == 0:
            print(f"Skipping fold. No training data before {first_test_date.date()}")
            continue

        # We can take a given forecaster and then fit it on our train data
        forecaster = forecaster_cls(config)
        forecaster.fit(X_train, y_train)
        y_pred = forecaster.predict(X_test)

        fold_results.append(
            {
                "dates": test_dates_block,
                "y_true": y_test,
                "y_pred": y_pred,
                "best_alpha": getattr(forecaster, "best_alpha_", None),
            }
        )

    print(f"walk_forward complete | folds={len(fold_results)}")
    return fold_results


def run_evaluation(fold_results):
    all_dates = np.concatenate([f["dates"] for f in fold_results])
    all_y_true = np.concatenate([f["y_true"] for f in fold_results])
    all_y_pred = np.concatenate([f["y_pred"] for f in fold_results])

    print(f"y_true mean={all_y_true.mean():.6f}, std={all_y_true.std():.6f}")
    print(f"y_pred mean={all_y_pred.mean():.6f}, std={all_y_pred.std():.6f}")

    metrics = compute_metrics(
        y_true=pd.Series(all_y_true, index=all_dates),
        y_pred=pd.Series(all_y_pred, index=all_dates),
    )

    print(f"IC={metrics['ic']:.4f} (p={metrics['p_value']:.4f})")
    print(f"DirAcc={metrics['diracc']:.4f}")
    print(f"MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

    return metrics
