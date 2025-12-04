from typing import Any, Union

import numpy as np


def T_total(
    G: int,
    N: Union[int, np.ndarray],
    L: int,
    W: int,
    p: Union[int, np.ndarray],
    Ccomm: float,
    alpha: float,
    beta: float,
    gamma: float,
    V: int
) -> np.ndarray:
    """
    Calculate the total time required for a neuroevolution process considering evaluation,
    reproduction, speciation, and communication times. The function integrates various
    parameters to compute the total time required for the given operation.

    :param G: The number of generations (epochs of evolution) to run.
    :param N: The number of individuals in the population.
    :param L: The number of steps in the environment simulation process to estimate the fitness of the individual.
    :param W: An average number of connections in the phenotype of each individual's ANN.
    :param p: The number of processing units available to parallelize part of the process.
    :param Ccomm: The communication time cost per individual in the population (seconds).
    :param alpha: A scaling factor for speciation time calculation (seconds per individual).
    :param beta: A scaling factor for reproduction time calculation (seconds per individual).
    :param gamma: A scaling factor for evaluation time calculation (seconds per individual).
    :param V: Additional computational weight tied to reproduction time.
    :return: The total time required for the process (seconds).
    """
    evaluation_time = N * L * W * gamma
    reproduction_time = beta * N * (W + V)
    speciation_time = alpha * N * np.log(N)
    communications_time = N * Ccomm

    return G * (evaluation_time / p + communications_time + speciation_time + reproduction_time)
