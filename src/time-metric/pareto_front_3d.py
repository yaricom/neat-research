import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from parameters import TimeMetricParameters, DOUBLE_POLE_BALANCING_PARAMETERS, PARETO_EPSILON
from time_metric import T_total


def draw_pareto_front_3d(params: TimeMetricParameters, show_title: bool = True) -> None:
    # Ranges
    N_vals = np.linspace(4, params.N, params.N-10)
    p_max = 64
    p_vals = np.linspace(1, p_max+1, p_max)

    N_grid, P_grid = np.meshgrid(N_vals, p_vals)
    mask = P_grid > N_grid

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
        V=params.V,
        delta=params.delta,
        kappa=params.kappa
    )
    T_grid = np.where(mask, np.nan, T_grid)

    # Per-N 5% near-optimal boundary:
    # For each N, find Tmin(N)=min_p T(N,p). Then choose the smallest p such that T(N,p) <= 1.05*Tmin(N).
    pareto_line_N = []
    pareto_line_p = []
    pareto_line_T = []

    for j, N_val in enumerate(N_vals):
        # valid p indices where p<=N
        valid_idx = np.where(p_vals <= N_val)[0]
        if len(valid_idx) == 0:
            continue
        T_col = T_grid[valid_idx, j]
        Tmin = np.nanmin(T_col)
        thr = (1.0 + PARETO_EPSILON) * Tmin

        # Find the smallest p meeting threshold
        ok_idx = valid_idx[np.where(T_grid[valid_idx, j] <= thr)[0]]
        if len(ok_idx) == 0:
            continue
        k = ok_idx[0]
        pareto_line_N.append(N_val)
        pareto_line_p.append(p_vals[k])
        pareto_line_T.append(T_grid[k, j])

    pareto_line_N = np.array(pareto_line_N)
    pareto_line_p = np.array(pareto_line_p)
    pareto_line_T = np.array(pareto_line_T)


    # Print statistics
    n_frontier = len(pareto_line_N)
    print(f"Near-Pareto frontier points: {n_frontier}")
    print(f"Min T value: {np.nanmin(T_grid):.2f}")
    print(f"Max T value: {np.nanmax(T_grid):.2f}")
    print(f"Pareto P at {N_val} value: {np.nanmax(pareto_line_p):.2f}")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    cmap = 'RdYlGn_r'
    ax.plot_surface(N_grid, P_grid, T_grid, cmap=cmap, alpha=0.8, linewidth=0, zorder=10)

    # Add contour lines
    zmin = np.nanmin(T_grid)
    ax.contour(N_grid, P_grid, T_grid,
               zdir='z',
               offset=zmin,
               cmap=cmap,
               levels=32)

    # Add per-N 5% boundary line
    ax.plot(pareto_line_N, pareto_line_p, pareto_line_T, color='cyan', linewidth=1,
            label=f'Per-N {PARETO_EPSILON*100:.0f}% near-optimal p boundary', zorder=9)

    ax.set_xlabel('Population size N')
    ax.set_ylabel('Workers number p (p <= N)')
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