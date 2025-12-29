from typing import Union

import numpy as np


def T_total(
    G: int,
    N: Union[int, np.ndarray],
    L: int,
    W: float,
    V: float,
    p: Union[int, np.ndarray],
    Ccomm: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    kappa: float
) -> np.ndarray:
    """
    Calculate the total time required for a neuroevolution process considering evaluation,
    reproduction, speciation, and communication times. The function integrates various
    parameters to compute the total time required for the given operation.

    :param G: The number of generations (epochs of evolution) to run.
    :param N: The number of individuals in the population.
    :param L: The number of steps in the environment simulation process to estimate the fitness of the individual.
    :param W: An average number of connections in the phenotype of each individual's ANN.
    :param V: An average number of nodes in the phenotype of each individual's ANN.
    :param p: The number of processing units available to parallelize part of the process.
    :param Ccomm: The communication time cost per individual in the population (seconds).
    :param alpha: A scaling factor for speciation time calculation (seconds per individual).
    :param beta: A scaling factor for reproduction time calculation (seconds per individual).
    :param gamma: A scaling factor for evaluation time calculation (seconds per individual).
    :param delta: A scaling factor for parallelism management time (seconds per individual).
    :param kappa: A degradation factor for sublinear increase of time for parallelism support (seconds per individual).

    :return: The total time required for the process (seconds).
    """
    evaluation = evaluation_time(N=N, L=L, V=V, gamma=gamma)
    reproduction = reproduction_time(beta=beta, N=N, W=W, V=V)
    speciation = speciation_time(alpha=alpha, N=N)
    communication = communication_time(N=N, Ccomm=Ccomm)
    parallelism_management = parallelism_management_time(p=1, delta=delta, kappa=kappa)

    return G * (evaluation / p + communication + speciation + reproduction + parallelism_management)


def non_parallelizable_time(
    N: int,
    W: float,
    V: float,
    Ccomm: float,
    alpha: float,
    beta: float,
    p: Union[int, np.ndarray],
    delta: float,
    kappa: float
) -> float:
    """
    Calculates the non-parallelizable time required by a system, considering reproduction,
    speciation, and communication times.

    The function combines the time required for reproduction, speciation, and communication
    to determine the overall non-parallelizable time. Each of these components is calculated
     using specific parameters and functions.

    :param N: The number of individuals in the population.
    :param W: An average number of connections in the phenotype of each individual's ANN.
    :param V: An average number of nodes in the phenotype of each individual's ANN.
    :param Ccomm: Communication cost or time per unit influencing communication time.
    :param alpha: Speciation-related parameter influencing speciation time.
    :param beta: Reproduction-related parameter influencing reproduction time.
    :return: The total non-parallelizable time as a float.
    """
    reproduction = reproduction_time(beta=beta, N=N, W=W, V=V)
    speciation = speciation_time(alpha=alpha, N=N)
    communication = communication_time(N=N, Ccomm=Ccomm)
    parallelism_management = parallelism_management_time(p=p, delta=delta, kappa=kappa)
    return reproduction + communication + speciation + parallelism_management


def evaluation_time(N: int, L: int, V: float, gamma: float) -> float:
    """
    Calculates the evaluation time of each individual in the population.

    This function computes the product of the input parameters `N`, `L`, `W`,
    and `gamma` to determine the evaluation time.

    :param N: The integer number indicating the number of tasks.
    :param L: The integer length parameter associated with each task.
    :param V: An average number of nodes in the phenotype of each individual's ANN.
    :param gamma: A scaling factor for evaluation time calculation (seconds per individual).
    :return: A float representing the computed evaluation time.
    """
    return N * L * V * gamma

def communication_time(N: int, Ccomm: float) -> float:
    return N * Ccomm

def speciation_time(N: int, alpha: float) -> float:
    return alpha * N * np.log(N)

def reproduction_time(N: int, beta: float, W: float, V: float) -> float:
    return beta * N * (W + V)

def parallelism_management_time(p: Union[int, np.ndarray], delta: float, kappa: float) -> float:
    return p * (delta ** kappa)