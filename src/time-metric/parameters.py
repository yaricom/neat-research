import dataclasses
from typing import Optional


@dataclasses.dataclass
class TimeMetricParameters:
    W: float
    V: float
    L: int
    G: int
    N: int
    Ccomm: float
    alpha: float
    beta: float
    gamma: float


DOUBLE_POLE_BALANCING_PARAMETERS = TimeMetricParameters(
    G=100, W=43.4, V=20.5, N=1_000, Ccomm=0.87e-8, alpha=1.89e-6, beta=0.98e-6, gamma=1.1e-9, L=100_000
)

PARETO_EPSILON = 0.05