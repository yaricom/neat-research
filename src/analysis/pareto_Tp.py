import os.path

import numpy as np
import matplotlib.pyplot as plt

from analysis.constants import OUTPUT_DIR
from analysis.model import time_model
from analysis.model_fit import load_fitted_model

fitted_model_df = load_fitted_model()

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

p_plot_max = 60
p_all = np.arange(1, p_plot_max + 1)

plt.figure(figsize=(10, 7))

for _, row in fitted_model_df.iterrows():
    N = int(row['N'])
    p = p_all[p_all <= N]
    T = time_model(p, A=row['A'], B=row['B'], delta=row['delta'], kappa=row['kappa'])
    Tmin = float(np.min(T))
    thr = 1.05 * Tmin # 5% from Tmin
    mask = T <= thr

    # curve
    (line,) = plt.plot(p, T, linewidth=2, marker='o', markersize=4, label=f"N={N}")
    # 5% near-optimal points
    color = line.get_color()
    plt.plot(p[mask], T[mask], linewidth=0, marker='o', markersize=6)

    # vertical line at minimum within the plotted range
    p_min = int(p[np.argmin(T)])
    plt.axvline(p_min, linestyle='--', linewidth=1, color=color)

plt.xlabel("Workers number $p$")
plt.ylabel("Time per epoch $T(p)$, ms")
plt.title("5% near Pareto-optimal $T(p)$ for different population sizes $N$")

plt.legend()
plt.tight_layout()

plt.show()

out = os.path.join(OUTPUT_DIR, "pareto5_Tp_model_curves.png")
plt.savefig(out, dpi=220)