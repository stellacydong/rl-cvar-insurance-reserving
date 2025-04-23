import pandas as pd
import numpy as np

def load_and_preprocess(path):
    df = pd.read_csv(path)
    # assume original columns: 'IncurredLosses'
    df["IncurredLosses_norm"] = df["IncurredLosses"] / df["IncurredLosses"].max()
    df["LocalVolatility"] = (
        df["IncurredLosses"]
        .rolling(window=10, min_periods=1)
        .std()
        .fillna(0)
        / df["IncurredLosses"].max()
    )
    return df

def compute_metrics(rollouts):
    """
    rollouts: list of dicts with keys R_t, L_t
    returns: RAR, CVaR_0.95, CES, RVR
    """
    R = np.array([x["R"] for x in rollouts])
    L = np.array([x["L"] for x in rollouts])
    shortfalls = np.maximum(0, L - R)

    # RAR
    RAR = np.mean(R / L)

    # CVaR_0.95
    var95 = np.quantile(shortfalls, 0.95)
    CVaR95 = np.mean(shortfalls[shortfalls >= var95])

    # CES
    CES = 1 - np.mean(np.abs(R - L))

    # RVR (assuming Rreg tracked in rollout)
    violations = np.array([x["violation"] for x in rollouts])
    RVR = violations.mean()

    return {"RAR": RAR, "CVaR95": CVaR95, "CES": CES, "RVR": RVR}
