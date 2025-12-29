# -*- coding: utf-8 -*-
"""
Saturation-model fit for T(p) curves (per epoch time) across multiple population sizes N.

Model (ms per epoch):
    T(N,p) = A_N / p + B_N + d0 * N * (p / (p + p_sat))

- A_N, B_N: per-N parameters
- p_sat, d0: global parameters shared across all N

Fitting method:
- Nonlinear least squares (scipy.optimize.least_squares)
- Unknown vector:
      theta = [log(p_sat), d0, A_100, B_100, A_500, B_500, ..., A_Nk, B_Nk]
  We optimize log(p_sat) to enforce p_sat > 0.
- Constraints:
      p_sat in [1, 10000]
      d0 >= 0
      A_N >= 0, B_N >= 0

Outputs:
- Fitted params table
- Per-N fit plots
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from analysis.constants import CSV_PATHS, OUTPUT_DIR
from analysis.metrics import rmse, r2_score
from analysis.model import T_sat


# ---------------------------
# 1) DATA LOADING
# ---------------------------


def _find_col(df: pd.DataFrame, candidates):
    """Return first matching column name (case-insensitive) from candidates, else None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def load_dataset(path: str) -> pd.DataFrame:
    """
    Expected:
      - workers/p column
      - time per epoch column in milliseconds (as user said)
    We try multiple common names.
    """
    df = pd.read_csv(path)

    # workers column
    col_p = _find_col(df, ["workers", "p", "num_workers", "n_workers"])
    if col_p is None:
        raise ValueError(f"Cannot find workers column in {path}. Columns: {list(df.columns)}")

    # time column (ms)
    col_t = _find_col(df, [
        "epoch_time_ms", "time_ms", "t_ms", "epoch_ms",
        "epoch time, ms", "time", "t"
    ])
    if col_t is None:
        raise ValueError(f"Cannot find time(ms) column in {path}. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "workers": df[col_p].astype(float),
        "epoch_time_ms": df[col_t].astype(float),
    })

    # Keep only p>=1 and sort by p
    out = out[out["workers"] >= 1].sort_values("workers").reset_index(drop=True)
    return out


datasets = {}
for N, path in CSV_PATHS.items():
    ds = load_dataset(path)
    datasets[N] = ds

Ns = sorted(datasets.keys())
idx = {N: i for i, N in enumerate(Ns)}


# ---------------------------
# 2) SATURATION MODEL
# ---------------------------


# ---------------------------
# 3) RESIDUALS FOR GLOBAL FIT
# ---------------------------

def residuals(theta: np.ndarray) -> np.ndarray:
    """
    theta = [log_p_sat, d0, A_0, B_0, A_1, B_1, ...]
    returns stacked residuals across all Ns.
    """
    log_p_sat, d0 = theta[0], theta[1]
    AB = theta[2:].reshape(len(Ns), 2)

    res = []
    for i, N in enumerate(Ns):
        A, B = AB[i]
        df = datasets[N]
        p = df["workers"].values
        y = df["epoch_time_ms"].values
        yhat = T_sat(N, p, A, B, log_p_sat, d0)
        res.append(yhat - y)
    return np.concatenate(res, axis=0)


# ---------------------------
# 4) INITIALIZATION
# ---------------------------

theta0 = []
# log_p_sat initial guess
theta0.append(np.log(200.0))     # p_sat ~ 200
# d0 initial guess (ms per organism). Start small.
theta0.append(1e-4)

# For each N: rough init A,B
# Use:
#   T(1) ~ A + B + d0*N*(1/(1+p_sat)) ~ A + B (since sat term small at p=1)
#   T(large p) ~ B + d0*N  (since p/(p+p_sat)->1 for p>>p_sat)
for N in Ns:
    df = datasets[N]
    # T at p=1 if present; else use first point
    if (df["workers"] == 1).any():
        T1 = float(df.loc[df["workers"] == 1, "epoch_time_ms"].iloc[0])
    else:
        T1 = float(df["epoch_time_ms"].iloc[0])

    Thigh = float(df["epoch_time_ms"].iloc[-1])

    # naive
    B0 = max(Thigh, 1e-9)   # ms
    A0 = max(T1 - B0, 1e-9) # ms
    theta0.extend([A0, B0])

theta0 = np.array(theta0, dtype=float)


# ---------------------------
# 5) BOUNDS + FIT (least squares)
# ---------------------------

lb = np.full_like(theta0, -np.inf)
ub = np.full_like(theta0,  np.inf)

# log_p_sat bounds: p_sat in [1, 10000]
lb[0] = np.log(1.0)
ub[0] = np.log(10000.0)

# d0 >= 0 (ms per organism)
lb[1] = 0.0
ub[1] = 1.0  # allow up to 1 ms/organism (wide enough)

# A_N, B_N >= 0
lb[2:] = 0.0

fit = least_squares(
    residuals,
    theta0,
    bounds=(lb, ub),
    max_nfev=80000,
)

if not fit.success:
    raise RuntimeError(f"Fit failed: {fit.message}")

theta = fit.x
log_p_sat, d0 = float(theta[0]), float(theta[1])
p_sat = float(np.exp(log_p_sat))
AB = theta[2:].reshape(len(Ns), 2)


# ---------------------------
# 6) METRICS + p* (analytic)
# ---------------------------

def p_star_model(N: float, A: float, p_sat: float, d0: float) -> float:
    """
    Analytical stationary point for:
        T = A/p + B + d0*N*(p/(p+p_sat))
    derivative:
        -A/p^2 + d0*N * p_sat/(p+p_sat)^2 = 0

    Solve for p:
        p* = (sqrt(A)*p_sat) / (sqrt(d0*N*p_sat) - sqrt(A))
    if denominator <= 0 => no finite minimizer (monotone decreasing on domain).
    """
    sqrtA = math.sqrt(max(A, 0.0))
    s = math.sqrt(max(d0 * N * p_sat, 0.0))
    denom = (s - sqrtA)
    if denom <= 0:
        return float("inf")
    return (sqrtA * p_sat) / denom


rows = []
for i, N in enumerate(Ns):
    A, B = float(AB[i, 0]), float(AB[i, 1])
    df = datasets[N]
    p = df["workers"].values
    y = df["epoch_time_ms"].values
    yhat = T_sat(N, p, A, B, log_p_sat, d0)
    rows.append({
        "N": N,
        "A_N (ms)": A,
        "B_N (ms)": B,
        "RMSE (ms)": rmse(y, yhat),
        "R^2": r2_score(y, yhat),
        "p*_model": p_star_model(N, A, p_sat, d0),
        "p_max_measured": float(np.max(p)),
        "T(p=1) ms": float(y[p.argmax() == 0]) if False else float(df.loc[df["workers"] == 1, "epoch_time_ms"].iloc[0]) if (df["workers"] == 1).any() else float(df["epoch_time_ms"].iloc[0]),
    })

summary = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

print("GLOBAL:")
print(f"  p_sat = {p_sat:.4f} workers")
print(f"  d0    = {d0:.8f} ms per organism (in term d0 * N * p/(p+p_sat))")
print("\nPER-N:")
print(summary.to_string(index=False))


# ---------------------------
# 7) PLOTS
# ---------------------------

out_dir = OUTPUT_DIR
os.makedirs(out_dir, exist_ok=True)

def plot_fit_for_N(N: int, A: float, B: float, p_sat: float, d0: float, savepath: str):
    df = datasets[N]
    p = df["workers"].values
    y = df["epoch_time_ms"].values

    pgrid = np.linspace(1, float(np.max(p)), 600)
    ygrid = T_sat(N, pgrid, A, B, np.log(p_sat), d0)

    plt.figure(figsize=(7, 4))
    plt.scatter(p, y, label="Експеримент", zorder=3)
    plt.plot(pgrid, ygrid, label="Saturation-модель", zorder=2)
    plt.xlabel("Кількість воркерів p")
    plt.ylabel("Час однієї епохи, мс")
    plt.title(f"T(p) з saturation-term (N={N})")
    plt.grid(True, alpha=0.3)

    # Mark p*_model if finite and within plot range
    pstar = p_star_model(N, A, p_sat, d0)
    if np.isfinite(pstar):
        plt.axvline(pstar, linestyle="--", linewidth=2, label=f"p*_model={pstar:.2f}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

# per-N plots
for i, N in enumerate(Ns):
    A, B = float(AB[i, 0]), float(AB[i, 1])
    savepath = os.path.join(out_dir, f"Tp_sat_fit_N{N}.png")
    plot_fit_for_N(N, A, B, p_sat, d0, savepath)

# combined plot
plt.figure(figsize=(9, 6))
for i, N in enumerate(Ns):
    df = datasets[N]
    p = df["workers"].values
    y = df["epoch_time_ms"].values
    A, B = float(AB[i, 0]), float(AB[i, 1])
    pgrid = np.linspace(1, float(np.max(p)), 600)
    ygrid = T_sat(N, pgrid, A, B, np.log(p_sat), d0)
    plt.scatter(p, y, s=20)
    plt.plot(pgrid, ygrid, label=f"N={N}")

plt.xlabel("Кількість воркерів p")
plt.ylabel("Час однієї епохи, мс")
plt.title("Saturation-модель: апроксимація T(p) для різних N")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
combined_path = os.path.join(out_dir, "Tp_sat_fit_combined.png")
plt.savefig(combined_path, dpi=200)
plt.close()

print(f"\nSaved combined plot: {combined_path}")
for N in Ns:
    print(f"Saved per-N plot: /mnt/data/Tp_sat_fit_N{N}.png")


# ---------------------------
# 8) EXPORT TABLE (optional)
# ---------------------------

table_path = os.path.join(out_dir, "saturation_fit_summary.csv")
summary.to_csv(table_path, index=False)
print(f"\nSaved summary table: {table_path}")