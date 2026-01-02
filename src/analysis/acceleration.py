from typing import List

import os
import pandas as pd

from analysis.constants import CSV_PATHS, DATA_DIR
from analysis.model_fit import load_fitted_model
from analysis.data_helpers import read_data_csv


def calculate_acceleration_efficiency(N_values: List[int]):
    """
    Calculate acceleration A(p) = T(1) / T(p) and efficiency E(p) = A(p) / p
    for the optimal number of workers across different population sizes.
    """
    # Load fitted model data with experimental minimums
    fitted_data = load_fitted_model()

    results = []

    for N in N_values:
        # Get experimental data for this N
        csv_path = CSV_PATHS[N]
        exp_data = read_data_csv(csv_path)
        exp_df = exp_data.df

        # Get T(1) - time with 1 worker
        t_1 = exp_df[exp_df[exp_data.p_column] == 1][exp_data.time_column].values[0]

        # Get optimal p and T(p*) from fitted model
        model_row = fitted_data[fitted_data['N'] == N].iloc[0]
        p_star = model_row['p_star_exp']
        t_p_star = model_row['t_min_exp']

        # Calculate acceleration: A(p) = T(1) / T(p)
        acceleration = t_1 / t_p_star

        # Calculate efficiency: E(p) = A(p) / p
        efficiency = acceleration / p_star

        results.append({
            'N': N,
            'p_star': int(p_star),
            'T(1)': t_1,
            'T(p*)': t_p_star,
            'A(p*)': acceleration,
            'E(p*)': efficiency
        })

        print(f"N={N}:")
        print(f"  p* = {int(p_star)}")
        print(f"  T(1) = {t_1:.3f} ms")
        print(f"  T(p*) = {t_p_star:.3f} ms")
        print(f"  Acceleration A(p*) = {acceleration:.3f}x")
        print(f"  Efficiency E(p*) = {efficiency:.3f}")
        print()

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_DIR, "acceleration_efficiency.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    calculate_acceleration_efficiency([500, 1000, 2000, 3000, 5000])