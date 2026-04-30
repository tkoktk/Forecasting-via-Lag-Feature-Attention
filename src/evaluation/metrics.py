import numpy as np
import pandas as pd
from scipy import stats


def compute_metrics(y_true, y_pred):
    df = pd.concat([y_true.rename("y"), y_pred.rename("yhat")], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        print("WARNING: No valid observations after alignment and NaN removal")
        return {"ic": np.nan, "diracc": np.nan, "mae": np.nan, "rmse": np.nan}

    # Our Information Coefficient ic = spearman(true, preds)
    ic, p_value = stats.spearmanr(df["y"], df["yhat"])
    diracc = float((np.sign(df["y"]) == np.sign(df["yhat"])).mean())
    mae = float(np.abs(df["y"] - df["yhat"]).mean())
    rmse = float(np.sqrt(((df["y"] - df["yhat"]) ** 2).mean()))

    print(
        f"IC={ic:.4f} (p={p_value:.4f}), DirAcc={diracc:.4f}, MAE={mae:.6f}, RMSE={rmse:.6f}"
    )

    return {"ic": ic, "p_value": p_value, "diracc": diracc, "mae": mae, "rmse": rmse}
