import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq

from analysis.charts import build_combined
from analysis.constants import DATA_DIR, OUTPUT_DIR, EXPERIMENTAL_WINDOW_SIZE
from analysis.model import estimate_pareto_band, time_model, p_star_exp_local_quadratic
from analysis.data_helpers import read_data_csv
from analysis.model_fit import load_fitted_model


def estimate_Tp_N(N_target: int, Ns: list[int]):
    # Load experimental data
    experiment_data_path = os.path.join(DATA_DIR, f"workers-time-{N_target}.csv")
    experimental_data = read_data_csv(experiment_data_path)

    p_exp = experimental_data.df["p"].to_numpy(dtype=float)
    T_exp = experimental_data.df["T_ms"].to_numpy(dtype=float)

    # Fitted model from calibration data
    calib = load_fitted_model(os.path.join(DATA_DIR, "analytical_fitted_model.csv"))

    # Filter out N values that need to be estimated (not measured)
    calib = calib[~calib["N"].isin(Ns)]

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
    p_sat_pred = float(coef_p_sat[0] + coef_p_sat[1]*lnNt)

    # Ensure positivity where needed
    A_pred = max(A_pred, 1e-12)
    d_pred = max(d_pred, 1e-12)

    # Predicted curve over the same p-range (continuous)
    points = 3000  # len(p_exp)  #
    p_grid = np.linspace(1, max(p_exp), points)
    T_pred = time_model(p=p_grid, A=A_pred, B=B_pred, delta=d_pred, p_sat=p_sat_pred)

    # Predicted optimum (continuous)
    idx_min = int(np.argmin(T_pred))
    p_model_star = float(p_grid[idx_min])
    T_model_star = float(T_pred[idx_min])

    # Predicted 5% Pareto band (in integer p for display + shading)
    p_band_min, p_band_max = estimate_pareto_band(
        p_exp_max=int(max(p_exp)), A_pred=A_pred, B_pred=B_pred, d_pred=d_pred, p_sat_pred=p_sat_pred
    )

    # Experimental optimum via local quadratic regression on 7 points around discrete minimum
    experimental_model = p_star_exp_local_quadratic(data=experimental_data, k=EXPERIMENTAL_WINDOW_SIZE)

    # Plot
    plt.figure(figsize=(9,5.5))
    plt.scatter(p_exp, T_exp, label=f"Experimental data (N={N_target})", zorder=3)

    plt.plot(p_grid, T_pred, "r--", linewidth=2, label="Estimated data (from calibration runs)")

    # Shade predicted 5% Pareto band
    if p_band_min is not None:
        plt.axvspan(p_band_min, p_band_max, alpha=0.12, label=f"Estimated 5% Pareto-area: p∈[{p_band_min},{p_band_max}]")

    # Vertical lines for optima
    plt.axvline(p_model_star, linestyle="--", linewidth=2, label=f"Estimated p*≈{p_model_star:.0f}")
    plt.axvline(experimental_model.p_star, linestyle=":", linewidth=2.5, label=f"Experimental p*≈{experimental_model.p_star:.0f} (quadratic by 7 point)")

    plt.xlim(p_exp.min(), 100)
    plt.xlabel(r"Workers number $p$")
    plt.ylabel("Time for one epoch of evolution, ms")
    plt.title(rf"N={N_target}: estimation $T(p)$ vs experiment + optimal and 5% near Pareto-optimal area")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    # plt.show()

    delta_T = math.fabs(T_model_star - experimental_model.t_min)
    epsilon = (delta_T / experimental_model.t_min) * 100
    pareto = 1.05 * T_model_star

    epsilon_p = math.fabs(p_model_star - experimental_model.p_star) / experimental_model.p_star * 100

    print(f"p* model: {p_model_star:.0f} (experimental: {experimental_model.p_star:.0f}), epsilon={epsilon_p:.1f}%")
    print(f"T* model: {T_model_star:.2f} (experimental: {experimental_model.t_min:.2f}), delta={delta_T:.3f}, epsilon={epsilon:.1f}% (pareto={pareto:.3f})")
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
        out_path = estimate_Tp_N(N, Ns=Ns)
        perN_paths.append(out_path)

    suffix = "-".join(map(str, Ns))
    combined_out = os.path.join(OUTPUT_DIR, f"combined_Tp_predictions-{suffix}.png")
    build_combined(perN_paths, combined_out, nrows=2, ncols=1, titles=["N=3000", "N=5000"], figsize=(8, 10))
    print("Saved combined figure:", combined_out)


if __name__ == "__main__":
    main()