import math
import os
from typing import Optional

import pandas as pd

from analysis.constants import OUTPUT_DIR, CSV_PATHS, DATA_DIR
from analysis.model import read_data_csv, fit_model_for_N, p_star_exp_local_quadratic, estimate_pareto_band


def load_fitted_model(csv_path: Optional[str] = None ):
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "analytical_fitted_model.csv")

    print(f"Loading fitted model from: {csv_path}")
    return pd.read_csv(csv_path)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data_rows = []

    for N, csv_path in CSV_PATHS.items():
        experimental_data = read_data_csv(csv_path)
        analytical_model = fit_model_for_N(N, data=experimental_data)
        experimental_model = p_star_exp_local_quadratic(data=experimental_data, k=7, p_col="p", t_col="T_ms")

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


if __name__ == "__main__":
    main()