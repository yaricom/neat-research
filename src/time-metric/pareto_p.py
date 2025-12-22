import numpy as np
import matplotlib.pyplot as plt

from parameters import TimeMetricParameters, DOUBLE_POLE_BALANCING_PARAMETERS, PARETO_EPSILON
from time_metric import p_knee, T_total

def draw_pareto_p(params: TimeMetricParameters, show_title: bool = True) -> None:
    p_vals=np.arange(1,64)

    T = T_total(
        G=params.G,
        N=params.N,
        L=params.L,
        W=params.W,
        V=params.V,
        p=p_vals,
        Ccomm=params.Ccomm,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma
    )

    # Heuristic Pareto zone: take points within 5% of minimal T
    T_min = T.min()
    threshold = T_min * (1.0 + PARETO_EPSILON)
    pareto_mask = T <= threshold

    print(f"T_min: {T_min:.3f}")
    p_recommended = p_vals[pareto_mask].min()
    print(f"Recommended p: {p_recommended:.1f} (minimal p within {PARETO_EPSILON*100:.0f}% of T_min)")
    p_knee_value = p_knee(
        N=params.N,
        L=params.L,
        W=params.W,
        V=params.V,
        Ccomm=params.Ccomm,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma
    )
    print(f"Performance knee: {p_knee_value:.2f}")

    plt.figure(figsize=(8,5))
    plt.plot(p_vals, T, marker='o', zorder=1, color='limegreen', label='All points')
    plt.scatter(p_vals[pareto_mask], T[pareto_mask], color='orange', label=f'Near-Pareto frontier ({PARETO_EPSILON*100}% of T_min)', s=30, zorder=2)

    plt.axhline(T_min, linestyle='--', label=f'T_min = {T_min:.3f} seconds', linewidth=1)
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
        DOUBLE_POLE_BALANCING_PARAMETERS, False
    )

if __name__ == "__main__":
    main(None)