import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from parameters import TimeMetricParameters, DOUBLE_POLE_BALANCING_PARAMETERS, PARETO_EPSILON
from time_metric import T_total


def draw_pareto_front_3d(params: TimeMetricParameters, show_title: bool = True) -> None:
    # Ranges
    N_vals = np.linspace(10, params.N, 990)
    p_vals = np.linspace(1, 64, 64)

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

    # Build near-Pareto frontier line for each N value
    frontier_N = []
    frontier_P = []
    frontier_T = []

    for col in range(T_grid.shape[1]):
        # Find minimum T for this N value
        T_min_col = T_grid[:, col].min()
        threshold_col = T_min_col * (1.0 + PARETO_EPSILON)

        # Get all near-Pareto points for this N
        near_pareto_indices = np.where(T_grid[:, col] <= threshold_col)[0]

        # Among near-Pareto points, find the one with minimum p (fewest workers)
        if len(near_pareto_indices) > 0:
            best_idx = near_pareto_indices[np.argmin(P_grid[near_pareto_indices, col])]
            frontier_N.append(N_grid[best_idx, col])
            frontier_P.append(P_grid[best_idx, col])
            frontier_T.append(T_grid[best_idx, col])

    frontier_N = np.array(frontier_N)
    frontier_P = np.array(frontier_P)
    frontier_T = np.array(frontier_T)

    # Print statistics
    n_frontier = len(frontier_N)
    print(f"Near-Pareto frontier points: {n_frontier}")
    print(f"Min T value: {T_grid.min():.2f}")
    print(f"Max T value: {T_grid.max():.2f}")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    cmap = 'RdYlGn_r'
    ax.plot_surface(N_grid, P_grid, T_grid, cmap=cmap, alpha=0.8, linewidth=0, zorder=10)

    # Add contour lines
    ax.contour(N_grid, P_grid, T_grid,
               levels=12,
               offset=T_grid.min()*0.95,
               cmap=cmap, alpha=0.5)

    # Plot near-Pareto frontier line
    ax.plot(frontier_N, frontier_P, frontier_T, '-', color='cyan', linewidth=1,
            label=f'Near-Pareto frontier ({PARETO_EPSILON*100:.0f}%)', zorder=9)

    ax.set_xlabel('Population size N')
    ax.set_ylabel('Workers number p')
    ax.set_zlabel('T_total (seconds)')
    if show_title:
        ax.set_title(f'3D поверхня T(N, p) з Near-Pareto фронтом\n(мінімізація часу T та кількості воркерів p)')

    plt.legend()
    plt.tight_layout()

    plt.show()


def main(args):
    draw_pareto_front_3d(
        DOUBLE_POLE_BALANCING_PARAMETERS, False
    )

if __name__ == "__main__":
    main(None)