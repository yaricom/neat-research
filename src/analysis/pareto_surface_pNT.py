import os.path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from analysis.constants import OUTPUT_DIR
from analysis.model import time_model
from analysis.model_fit import load_fitted_model

# Fitted params at discrete N
data = load_fitted_model()
logN = np.log(data["N"].values.astype(float))


def interp_param(param_name, N_query):
    y = data[param_name].values.astype(float)
    return np.interp(np.log(N_query), logN, y, left=y[0], right=y[-1])


def T_model(N, p):
    # G=1 scaling (doesn't affect argmin/pareto band)
    A = interp_param("A", N)
    B = interp_param("B", N)
    delta = interp_param("delta", N)
    kappa = interp_param("kappa", N)
    return time_model(
        p, A=A, B=B, delta=delta, kappa=kappa
    )


# Grid (only plot up to p_max)
N_vals = np.linspace(100, 2000, 140)
p_max = 100
p_vals = np.arange(1, p_max + 1)

NN, PP = np.meshgrid(N_vals, p_vals, indexing="xy")

# Mask invalid region p > N
mask_invalid = PP > NN

A = interp_param("A", NN)
B = interp_param("B", NN)
delta = interp_param("delta", NN)
kappa = interp_param("kappa", NN)

T = (A / PP) + B + delta * (PP ** kappa)
T = np.where(mask_invalid, np.nan, T)

# Compute 5% near-Pareto band per N (based on Tmin(N) over allowed p)
p_low = np.empty_like(N_vals)
p_high = np.empty_like(N_vals)
T_low = np.empty_like(N_vals)
T_high = np.empty_like(N_vals)
p_star = np.empty_like(N_vals)
T_star = np.empty_like(N_vals)

for i, N in enumerate(N_vals):
    p_allowed_max = int(min(p_max, np.floor(N)))
    ps = np.arange(1, p_allowed_max + 1)
    Ts = T_model(N, ps)
    Tmin = Ts.min()
    thresh = 1.05 * Tmin  # +5% from Tmin

    # Define a band as all p achieving within 5% of Tmin(N)
    ok = Ts <= thresh
    # Safety in case of numerical issues
    if not np.any(ok):
        idx = int(np.argmin(Ts))
        p_low[i] = p_high[i] = ps[idx]
    else:
        p_low[i] = ps[np.argmax(ok)]  # first True
        p_high[i] = ps[len(ok) - 1 - np.argmax(ok[::-1])]  # last True

    # Store corresponding T
    T_low[i] = T_model(N, p_low[i])
    T_high[i] = T_model(N, p_high[i])

    # Continuous optimum (clipped)
    Ai = interp_param("A", N)
    di = interp_param("delta", N)
    ki = interp_param("kappa", N)
    p_cont = (Ai / (di * ki)) ** (1.0 / (ki + 1.0))
    p_star[i] = np.clip(p_cont, 1, p_allowed_max)
    T_star[i] = T_model(N, p_star[i])

# Plot
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

# Surface (downsample for speed)
cmap = 'RdYlGn_r'
surf = ax.plot_surface(NN, PP, T, rstride=2, cmap=cmap, cstride=2, linewidth=0, antialiased=True, alpha=0.8, zorder=5)

# Overlay optimal line and 5% band boundaries
ax.plot(
    N_vals, p_star, T_star, linewidth=2, color="cyan", label=r"$p^*$ model", zorder=6
)
ax.plot(
    N_vals, p_low, T_low, linewidth=2, color="orange", label=r"$p_{min}^{5\%}$ at 5% from $T_{min}$", zorder=6
)
# ax.plot(N_vals, p_high, T_high, linewidth=2, color="green", zorder=6)

# Contours on the bottom plane for readability (only finite values)
finite_T = np.nan_to_num(T, nan=np.nanmin(T[np.isfinite(T)]))
zmin = np.nanmin(T[np.isfinite(T)])
ax.contour(NN, PP, finite_T, zdir="z", cmap=cmap, offset=zmin, levels=10)

ax.set_xlabel("Population size N")
ax.set_ylabel("Workers p (constraint: p ≤ N)")
ax.set_zlabel("Time per generation T(N,p), ms")
ax.set_title("T(N,p) in region with per-N 5% near Pareto-optimal frontier")

# Limit z to reduce distortion from outliers
ax.set_zlim(zmin, np.nanpercentile(T[np.isfinite(T)], 99))

plt.legend()
plt.tight_layout()
fig.show()


out_path = os.path.join(OUTPUT_DIR, "pNT_surface_with_pareto5_band.png")
plt.savefig(out_path, dpi=220)