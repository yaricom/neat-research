import dataclasses
import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analysis.metrics import r2_score

# --- optional LOWESS ---
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False


@dataclasses.dataclass
class ExperimentalModelFit:
    p_star: float
    a: float
    b: float
    c: float
    t_min: float
    w: pd.DataFrame


@dataclasses.dataclass
class AnalyticModelFit:
    N: int
    A: float
    B: float
    delta: float
    kappa: float
    p_star_model: float
    t_min: float
    p_sat: float
    # coefficient of determination
    r2: float


@dataclasses.dataclass
class ExperimentalData:
    df: pd.DataFrame
    efficiency_score_column: str
    time_column: str
    p_column: str


def time_model(p, A, B, delta, p_sat):
    """
    The analytical simplified saturation time model
    T(N,p) = A/p + B + delta*(p/(p+p_sat))
    """
    p = np.asarray(p, dtype=float)
    return A/p + B + delta*(p/(p + p_sat))


def fit_model_for_N(N: int, data: ExperimentalData):
    """Load CSV, infer columns, fit model parameters, return a dict with everything needed for plotting."""
    p = data.df["p"].values.astype(float)
    T = data.df["T_ms"].values.astype(float)

    # initial guesses
    B0 = float(np.min(T))
    A0 = float(max(1e-9, (T[0] - B0) * p[0]))
    delta0 = float(max(1e-9, (T[-1] - B0) / max(1.0, p[-1] ** 0.5)))
    ps0 = np.median(p)

    popt, _ = curve_fit(
        time_model, p, T,
        p0=[A0, B0, delta0, ps0],
        bounds=(0, np.inf),
        maxfev=200000
    )
    A, B, delta, p_sat = popt

    if delta > 0 and p_sat > 0:
        p_star_m = p_star_model_p_sat(A=A, delta=delta, p_sat=p_sat)
    else:
        p_star_m = float("nan")

    # coefficient of determination R^2
    T_hat = time_model(p, A, B, delta, p_sat)
    r2 = r2_score(T, T_hat)

    T_min = float(np.nanmin(T_hat))

    return AnalyticModelFit(
        N=N, A=A, B=B, delta=delta, kappa=0, p_star_model=p_star_m, t_min=T_min, r2=r2, p_sat=p_sat
    )

def p_star_model_p_sat(A: float, delta: float, p_sat: float):
    """
    Calculate the optimal number of workers p*_model for given saturation time model parameters.
    """
    return p_sat / (math.sqrt(delta * p_sat / A) - 1.0)


def p_star_exp_local_quadratic(data: ExperimentalData, k: int = 7):
    """
    Local estimation p*_exp with quadratic approximation.
    T(p) = a p^2 + b p + c for k the closest points around minimum.

    Parameters:
      data: the experimental data
      k: number of points in the local window (recommended 5..9, odd)
    Returns:
      p_star (float), coeffs (a,b,c), window_df (local window of points)
    """
    d = data.df[[data.p_column, data.time_column]].dropna().copy()
    d = d.sort_values(data.p_column)
    T = d[data.time_column].to_numpy(dtype=float)

    if len(d) < 3:
        raise ValueError("Required at least 3 points for quadratic approximation of p*.")
    if k < 3:
        k = 3
    if k > len(d):
        k = len(d)

    # index of the experimental minimum
    i0 = int(np.argmin(T))

    # local window with k points around i0
    half = k // 2
    left = max(0, i0 - half)
    right = min(len(d), left + k)
    left = max(0, right - k)

    w = d.iloc[left:right].copy()
    pw = w[data.p_column].to_numpy(dtype=float)
    Tw = w[data.time_column].to_numpy(dtype=float)

    # quadratic approximation
    # coeffs: [a, b, c] для a p^2 + b p + c
    a, b, c = np.polyfit(pw, Tw, 2)

    # if parabola is looking down, or linear - found minimum is unstable
    if a <= 1e-18:
        # fallback: taking minimum among points in the window
        p_star = float(pw[np.argmin(Tw)])
        return ExperimentalModelFit(p_star=p_star, a=a, b=b, c=c, w=w, t_min=float(np.nanmin(Tw)))

    p_star = float(-b / (2.0 * a))
    t_min = float((4*a*c) - b**2)/float(4*a)

    # limiting p* in the range of a local interval to avoid drifting too much
    p_star = float(np.clip(p_star, pw.min(), pw.max()))

    return ExperimentalModelFit(p_star=p_star, a=a, b=b, c=c, w=w, t_min=t_min)


# -------------------------
# A2) LOWESS (optional)
# -------------------------
def p_star_exp_lowess(df: pd.DataFrame, p_col="p", t_col="T_ms",
                      frac=0.25, it=3, grid_size=5000):
    """
    Nonparametric LOWESS smoothing: T_hat(p) via LOWESS, then p* = argmin.
    Requires: statsmodels
    Parameters:
      frac: smoothing span (0..1), larger => smoother
      it: robustness iterations
    Returns:
      p_star, (p_smooth, T_smooth), (p_grid, T_grid)
    """
    if not HAS_LOWESS:
        raise RuntimeError("statsmodels is not installed; install it or use spline method.")

    d = df[[p_col, t_col]].dropna().copy().sort_values(p_col)
    p = d[p_col].to_numpy(float)
    T = d[t_col].to_numpy(float)

    # LOWESS (returns ordered pairs)
    sm = lowess(T, p, frac=frac, it=it, return_sorted=True)
    p_s, T_s = sm[:, 0], sm[:, 1]

    # Dense search for a minimum
    p_grid = np.linspace(p.min(), p.max(), grid_size)
    T_grid = np.interp(p_grid, p_s, T_s)

    i = int(np.argmin(T_grid))
    p_star = float(p_grid[i])

    return p_star, (p_s, T_s), (p_grid, T_grid)


def estimate_pareto_band(
    p_exp_max: int, A_pred: float, B_pred: float, d_pred: float, p_sat_pred: float
) -> Tuple[int, int]:
    p_int = np.arange(1, p_exp_max + 1) #  int(max(p_exp))
    T_pred_int = time_model(
        p_int,
        A=A_pred,
        B=B_pred,
        delta=d_pred,
        p_sat=p_sat_pred
    )
    Tmin_int = float(T_pred_int.min())
    thr = 1.05 * Tmin_int
    band = p_int[T_pred_int <= thr]
    p_band_min = int(band.min()) if len(band) else None
    p_band_max = int(band.max()) if len(band) else None
    return p_band_min, p_band_max