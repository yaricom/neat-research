import numpy as np


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """Computes coefficient of determination; handles a degenerate total sum"""
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error"""
    return float(np.sqrt(np.mean((yhat - y) ** 2)))