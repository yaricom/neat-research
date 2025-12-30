import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

from analysis.charts import build_combined
from analysis.constants import DATA_DIR, OUTPUT_DIR
from analysis.model import read_data_csv, estimate_pareto_band
from analysis.model_fit import load_fitted_model


def estimate_Tp_N(N_target):
    # Load experimental data
    df_to_predict = read_data_csv(os.path.join(DATA_DIR, f"workers-time-{N_target}.csv")).df
    p_exp = df_to_predict["p"].to_numpy(dtype=float)
    T_exp = df_to_predict["T_ms"].to_numpy(dtype=float)

    # Fitted model from calibration data
    calib = load_fitted_model(os.path.join(DATA_DIR, "analytical_fitted_model-500_3000.csv"))

    # Fit simple models of parameters vs log(N): param = a + b*logN
    logN = np.log(calib["N"].values)
    def fit_lin(x, y):
        X = np.column_stack([np.ones_like(x), x])
        coef, *_ = lstsq(X, y, rcond=None)
        return coef  # [a,b]

    coef_A = fit_lin(logN, calib["A"].values)
    coef_B = fit_lin(logN, calib["B"].values)
    coef_d = fit_lin(logN, calib["delta"].values)
    coef_p_sat = fit_lin(logN, calib["p_sat"].values)

    lnNt = np.log(N_target)
    A_pred = float(coef_A[0] + coef_A[1]*lnNt)
    B_pred = float(coef_B[0] + coef_B[1]*lnNt)
    d_pred = float(coef_d[0] + coef_d[1]*lnNt)
    k_pred = float(coef_k[0] + coef_k[1]*lnNt)

    # Ensure positivity where needed
    A_pred = max(A_pred, 1e-12)
    d_pred = max(d_pred, 1e-12)
    k_pred = max(k_pred, 1e-6)

    # Predicted curve over the same p-range (continuous)
    points = 800  # len(p_exp)  #
    p_grid = np.linspace(1, max(p_exp), points)
    T_pred = A_pred/p_grid + B_pred + d_pred*(p_grid**k_pred)

    # Predicted optimum (continuous)
    idx_min = int(np.argmin(T_pred))
    p_model_star = float(p_grid[idx_min])
    T_model_star = float(T_pred[idx_min])

    # Predicted 5% Pareto band (in integer p for display + shading)
    p_band_min, p_band_max = estimate_pareto_band(
        p_exp_max=int(max(p_exp)), A_pred=A_pred, B_pred=B_pred, d_pred=d_pred, k_pred=k_pred
    )

    # Experimental optimum via local quadratic regression on 7 points around discrete minimum
    order = np.argsort(p_exp)
    p_sorted = p_exp[order]
    T_sorted = T_exp[order]

    i0 = int(np.argmin(T_sorted))  # discrete min index in sorted array
    # pick 7 points centered at i0 (clamped)
    start = max(0, i0-3)
    end = min(len(p_sorted), start+7)
    start = max(0, end-7)
    p7 = p_sorted[start:end]
    T7 = T_sorted[start:end]

    # quadratic fit: T = a p^2 + b p + c
    X = np.column_stack([p7**2, p7, np.ones_like(p7)])
    (a,b,c), *_ = lstsq(X, T7, rcond=None)

    # vertex
    if a > 0:
        # from derivative = 0 at vertex: -2b/a
        p_exp_star = -b/(2*a)
    else:
        p_exp_star = float(p_sorted[i0])
    # clamp to possible domain
    p_exp_star = float(np.clip(p_exp_star, p_sorted.min(), p_sorted.max()))
    # the experimental minimum (4*a*c - b^2)/4*a
    T_exp_star = float((4*a*c) - b**2)/float(4*a)

    # Plot
    plt.figure(figsize=(9,5.5))
    plt.scatter(p_exp, T_exp, label=f"Experimental data (N={N_target})", zorder=3)

    plt.plot(p_grid, T_pred, "r--", linewidth=2, label="Estimated data (from calibration runs)")

    # Shade predicted 5% Pareto band
    if p_band_min is not None:
        plt.axvspan(p_band_min, p_band_max, alpha=0.12, label=f"Estimated 5% Pareto-area: p∈[{p_band_min},{p_band_max}]")

    # Vertical lines for optima
    plt.axvline(p_model_star, linestyle="--", linewidth=2, label=f"Estimated p*≈{p_model_star:.0f}")
    plt.axvline(p_exp_star, linestyle=":", linewidth=2.5, label=f"Experimental p*≈{p_exp_star:.0f} (quadratic by 7 point)")

    plt.xlabel(r"Workers number $p$")
    plt.ylabel("Time for one epoch of evolution, ms")
    plt.title(rf"N={N_target}: estimation $T(p)$ vs experiment + optimal and 5% near Pareto-optimal area")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    delta_T = math.fabs(T_model_star - T_exp_star)
    epsilon = (delta_T / T_exp_star) * 100
    pareto = 1.05 * T_model_star

    epsilon_p = math.fabs(p_model_star - p_exp_star) / p_exp_star * 100

    print(f"p* model: {p_model_star:.0f} (experimental: {p_exp_star:.0f}), epsilon={epsilon_p:.1f}%")
    print(f"T* model: {T_model_star:.2f} (experimental: {T_exp_star:.2f}), delta={delta_T:.3f}, epsilon={epsilon:.1f}% (pareto={pareto:.3f})")
    print(f"5% near-optimal band: [{p_band_min},{p_band_max}]")


    out_path = os.path.join(OUTPUT_DIR, f"Tp_N{N_target}_pred_vs_exp_pareto5_optima.png")
    plt.savefig(out_path, dpi=200)

    print(f"Saved to: {out_path}")
    return out_path


def main():
    perN_paths = []

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    Ns = [3000, 5000]
    for N in Ns:
        out_path = estimate_Tp_N(N)
        perN_paths.append(out_path)

    suffix = "-".join(map(str, Ns))
    combined_out = os.path.join(OUTPUT_DIR, f"combined_Tp_predictions-{suffix}.png")
    build_combined(perN_paths, combined_out, nrows=2, ncols=1, titles=["N=3000", "N=5000"], figsize=(8, 10))
    print("Saved combined figure:", combined_out)


if __name__ == "__main__":
    main()