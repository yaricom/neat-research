import numpy as np
import matplotlib.pyplot as plt

from parameters import TimeMetricParameters, DOUBLE_POLE_BALANCING_PARAMETERS, PARETO_EPSILON
from time_metric import T_total

def render_pareto_front(params: TimeMetricParameters, show_title: bool = True) -> None:
    # Grid
    N_vals = np.linspace(50, params.N, 80)
    p_vals = np.linspace(1, 65, 64)
    N_grid, P_grid = np.meshgrid(N_vals, p_vals)

    T_grid = T_total(
        G=params.G,
        N=N_grid,
        L=params.L,
        W=params.W,
        p=P_grid,
        Ccomm=params.Ccomm,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        V=params.V
    )

    # Flatten
    N_flat = N_grid.flatten()
    P_flat = P_grid.flatten()
    T_flat = T_grid.flatten()

    # Build near-Pareto frontier line for two objectives: minimize T and minimize p (cost of workers)
    # For each N value, find the optimal p value (minimum T)
    frontier_N = []
    frontier_P = []

    for col in range(T_grid.shape[1]):
        # Find a minimum T for this N value
        T_min_col = T_grid[:, col].min()
        threshold_col = T_min_col * (1.0 + PARETO_EPSILON)

        # Get all near-Pareto points for this N
        near_pareto_indices = np.where(T_grid[:, col] <= threshold_col)[0]

        # Among near-Pareto points, find the one with minimum p (fewest workers)
        if len(near_pareto_indices) > 0:
            best_idx = near_pareto_indices[np.argmin(P_grid[near_pareto_indices, col])]
            frontier_N.append(N_grid[best_idx, col])
            frontier_P.append(P_grid[best_idx, col])

    frontier_N = np.array(frontier_N)
    frontier_P = np.array(frontier_P)

    # Print statistics
    n_frontier = len(frontier_N)
    n_total = len(T_flat)
    print(f"Total points: {n_total}")
    print(f"Near-Pareto frontier points: {n_frontier}")
    print(f"Min T value: {T_flat.min():.2f}")
    print(f"Max T value: {T_flat.max():.2f}")

    # Plot all points colored by T
    plt.figure(figsize=(8,6))
    sc = plt.scatter(N_flat, P_flat, c=T_flat, cmap='RdYlGn_r', alpha=0.3)
    plt.colorbar(sc, label="T_total (seconds)")

    # Plot near-Pareto frontier line
    plt.plot(frontier_N, frontier_P, 'b-', linewidth=2, label=f'Near-Pareto frontier ({PARETO_EPSILON*100:.0f}%)', zorder=3)
    # plt.scatter(frontier_N, frontier_P, c='blue', s=20, marker='o', edgecolors='darkblue', linewidths=2, zorder=4)

    plt.xlabel("Population size N")
    plt.ylabel("Workers count p")
    if show_title:
        plt.title("Pareto-фронт у просторі параметрів (N, p)\n(мінімізація часу T та кількості воркерів p)")
    plt.legend()
    plt.tight_layout()

    plt.show()


def main(args):
    render_pareto_front(
        DOUBLE_POLE_BALANCING_PARAMETERS
    )


if __name__ == "__main__":
    main(None)