import dataclasses


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
    delta: float
    kappa: float


DOUBLE_POLE_BALANCING_PARAMETERS = TimeMetricParameters(
    G=1, W=26, V=12, N=2_000, Ccomm=1.1e-9,
    alpha=1.07e-7, beta=5.5e-9, gamma=6.35e-10,
    delta=0.8345, kappa=0.6995,
    L=100_000
)

PARETO_EPSILON = 0.05