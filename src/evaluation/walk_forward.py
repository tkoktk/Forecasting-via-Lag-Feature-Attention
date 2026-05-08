import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_metrics


def walk_forward(
    X, y, dates, forecaster_cls, config, splitter, inner_splitter=None, verbose=False
):
    dates = pd.DatetimeIndex(dates)
    fold_results = []

    for fold in splitter.split(dates):
        X_train = X[fold.train_idx]
        y_train = y[fold.train_idx]
        X_test = X[fold.test_idx]
        y_test = y[fold.test_idx]
        test_dates_block = dates[fold.test_idx]

        if verbose:
            print(
                f"fold {fold.fold_id} | n_train={fold.n_train}, n_test={fold.n_test}, "
                f"n_purged={fold.n_purged} | test_start={fold.test_start_date.date()}"
            )

        forecaster = forecaster_cls(config)

        if inner_splitter is None:
            forecaster.fit(X_train, y_train)

        else:  # If we are using a model that uses early stopping:
            inner_train_idx, val_idx = inner_splitter.split(fold.n_train)
            forecaster.fit(
                X_train[inner_train_idx],
                y_train[inner_train_idx],
                X_val=X_train[val_idx],
                y_val=y_train[val_idx],
            )

        y_pred = forecaster.predict(X_test)

        fold_results.append(
            {
                "fold_id": fold.fold_id,
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
