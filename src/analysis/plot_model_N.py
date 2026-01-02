"""
Build final T(p) plots for each population size N from CSV files:
- scatter/line of experimental times (ms)
- fitted curve: T(p) = A/p + B + δ p^κ
- vertical lines for p*_exp (min from data) and p*_model (analytic optimum)
- saves one PNG per N and one combined 2x2 figure

Requirements:
  pip install pandas numpy matplotlib scipy pillow

CSV expectations:
  - one column with workers count (p)
  - one column with time in milliseconds (any name containing "time")
  - optional column "EfficiencyScore" (not required for these plots)
"""
import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.charts import build_combined
from analysis.constants import CSV_PATHS, OUTPUT_DIR, DATA_DIR, EXPERIMENTAL_WINDOW_SIZE
from analysis.model import AnalyticModelFit, ExperimentalData, ExperimentalModelFit, fit_model_for_N, \
    p_star_exp_local_quadratic, time_model, estimate_pareto_band
from analysis.data_helpers import read_data_csv


def plot_one(
    analytic_model: AnalyticModelFit,
    experimental_model: ExperimentalModelFit,
    experimental_data: ExperimentalData,
    out_path: str
) -> None:
    """Plot one N figure and save."""
    d = experimental_data.df
    p = d["p"].values.astype(float)
    T = d["T_ms"].values.astype(float)

    # dense grid for fitted curve
    p_stop = 90  # p.max()
    p_grid = np.linspace(p.min(), p_stop, 400)
    T_fit = time_model(p=p_grid, A=analytic_model.A, B=analytic_model.B, delta=analytic_model.delta, p_sat=analytic_model.p_sat)

    # Predicted 5% Pareto band (in integer p for display + shading)
    p_band_min, p_band_max = estimate_pareto_band(
        p_exp_max=int(max(p)), A_pred=analytic_model.A, B_pred=analytic_model.B, d_pred=analytic_model.delta, p_sat_pred=analytic_model.p_sat
    )

    plt.figure()
    plt.plot(p, T, marker="o", linestyle="-", label="Experimental data")
    plt.plot(
        p_grid, T_fit, linestyle="--",
        label=r"Model $A/p + B + \delta\frac{p}{p+p_{\mathrm{sat}}}$ (R$^2$=%.4f)" % analytic_model.r2
    )

    # T(p)=\frac{A}{p}+B+\delta\,\frac{p}{p+p_{\mathrm{sat}}},

    # Shade predicted 5% Pareto band
    if p_band_min is not None:
        plt.axvspan(p_band_min, p_band_max, alpha=0.12,
                    label=f"Estimated 5% Pareto-area: p∈[{p_band_min},{p_band_max}]")

    # local quadratic approximation of p*_exp (plot only in fitted window range)
    w = experimental_model.w
    p_window_min = w["p"].min()
    p_window_max = w["p"].max()
    p_loc = np.linspace(p_window_min, p_window_max, num=200)
    T_loc = experimental_model.a * p_loc ** 2 + experimental_model.b * p_loc + experimental_model.c
    plt.plot(p_loc, T_loc, linestyle=":", linewidth=2, label="Local quadratic approximation ($p^*_{exp}$)")

    plt.axvline(experimental_model.p_star, linestyle=":", label=rf"$p^*_{{exp,loc}}={experimental_model.p_star:.0f}$")
    plt.axvline(analytic_model.p_star_model, linestyle="-.",
                label=rf"$p^*_{{model}}={analytic_model.p_star_model:.0f}$")

    plt.xlim(p.min(), p_stop)
    plt.xlabel(r"Workers number $p$")
    plt.ylabel("Time for one epoch of evolution, ms")
    # plt.title(f"Залежність часу від кількості воркерів (N={analytic_model.N})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    results = {}
    perN_paths = []

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data_rows = []

    for N, csv_path in CSV_PATHS.items():
        experimental_data = read_data_csv(csv_path)
        analytical_model = fit_model_for_N(N, data=experimental_data)
        results[N] = analytical_model
        experimental_model= p_star_exp_local_quadratic(data=experimental_data, k=EXPERIMENTAL_WINDOW_SIZE)

        out = os.path.join(OUTPUT_DIR, f"final_Tp_N{N}.png")
        plot_one(
            analytic_model=analytical_model,
            experimental_model=experimental_model,
            experimental_data=experimental_data,
            out_path=out
        )
        perN_paths.append(out)

        delta_T = math.fabs(analytical_model.t_min - experimental_model.t_min)
        epsilon_T = (delta_T / experimental_model.t_min) * 100
        pareto = 1.05 * analytical_model.t_min

        p_star_model = round(analytical_model.p_star_model, 0)
        p_star_exp = round(experimental_model.p_star, 0)
        epsilon_p = math.fabs(p_star_model - p_star_exp) / p_star_exp * 100

        p_band_min, p_band_max = estimate_pareto_band(
            p_exp_max=int(max(experimental_data.df["p"])),
            A_pred=analytical_model.A,
            B_pred=analytical_model.B,
            d_pred=analytical_model.delta,
            p_sat_pred=analytical_model.p_sat,
        )

        # Save data to dataframe
        data_rows.append({
            'N': N,
            'p_star_exp': int(p_star_exp),
            'p_star_model': int(p_star_model),
            'epsilon_p': epsilon_p,
            't_min_exp': experimental_model.t_min,
            't_min_model': analytical_model.t_min,
            'delta_T': delta_T,
            'epsilon_T': epsilon_T,
            'pareto_5%': pareto,
            'p_band_5%_min': p_band_min,
            'p_band_5%_max': p_band_max,
            'A': analytical_model.A,
            'B': analytical_model.B,
            'delta': analytical_model.delta,
            'R^2': analytical_model.r2,
            'p_sat': analytical_model.p_sat,
        })

        print(
            f"N={N}: p*_exp={p_star_exp:.0f}, p*_model={p_star_model:.0f}, epsilon={epsilon_p:.1f}, p_saturation={analytical_model.p_sat:.0f}, R^2={analytical_model.r2:.4f}, pareto band: [{p_band_min},{p_band_max}]")
        print(
            f"  T_min_exp: {experimental_model.t_min:.3f}, T_min_model: {analytical_model.t_min:.3f}, dT={delta_T:.3f}, epsilon={epsilon_T:.1f}, pareto={pareto:.3f}")
        print(
            f"  params: A={analytical_model.A:.6g}, B={analytical_model.B:.6g}, delta={analytical_model.delta:.6g}, kappa={analytical_model.kappa:.6g}")
        print("\n")

    # Create a dataframe from collected data and save to CSV
    data_df = pd.DataFrame(data_rows)
    data_csv_path = os.path.join(DATA_DIR, "analytical_fitted_model.csv")
    data_df.to_csv(data_csv_path, index=False)
    print(f"Saved data to: {data_csv_path}\n")

    combined_out = os.path.join(OUTPUT_DIR, "final_combined_Tp.png")
    build_combined(perN_paths, combined_out, nrows=2, ncols=2, titles=["N=500", "N=1000", "N=2000", "N=3000", ])
    print("Saved combined figure:", combined_out)

if __name__ == "__main__":
    main()