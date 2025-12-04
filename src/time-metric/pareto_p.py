import numpy as np
import matplotlib.pyplot as plt

from parameters import TimeMetricParameters, DOUBLE_POLE_BALANCING_PARAMETERS, PARETO_EPSILON
from time_metric import T_total

def draw_pareto_p(params: TimeMetricParameters, show_title: bool = True) -> None:
    p_vals=np.arange(1,64)

    T = T_total(
        G=params.G,
        N=params.N,
        L=params.L,
        W=params.W,
        p=p_vals,
        Ccomm=params.Ccomm,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        V=params.V
    )

    # Heuristic Pareto zone: take points within 5% of minimal T
    T_min = T.min()
    threshold = T_min * (1.0 + PARETO_EPSILON)
    pareto_mask = T <= threshold

    plt.figure(figsize=(8,5))
    plt.plot(p_vals, T, marker='o', zorder=1, color='green')
    plt.scatter(p_vals[pareto_mask], T[pareto_mask], color='blue', label=f'Near-Pareto frontier ({PARETO_EPSILON*100}% of T_min)', s=30, zorder=2)

    plt.axhline(T_min, linestyle='--', label='T_min', linewidth=1)
    plt.xlabel("Workers count p")
    plt.ylabel("T_total (seconds)")
    if show_title:
        plt.title(f"Pareto-подібний вибір (p) з виділеною оптимальною зоною (якщо N = {params.N})")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.show()


def main(args):
    draw_pareto_p(
        DOUBLE_POLE_BALANCING_PARAMETERS
    )

if __name__ == "__main__":
    main(None)