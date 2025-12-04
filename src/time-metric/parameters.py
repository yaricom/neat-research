import dataclasses
from typing import Optional


@dataclasses.dataclass
class TimeMetricParameters:
    G: int
    W: int
    N: int
    Ccomm: float
    alpha: float
    beta: float
    gamma: float
    V: int
    L: int


DOUBLE_POLE_BALANCING_PARAMETERS = TimeMetricParameters(
    G=50, W=9, N=1000, Ccomm=0.00002, alpha=1e-3, beta=5e-4, gamma=0.00001, V=2, L=200
)

PARETO_EPSILON = 0.05