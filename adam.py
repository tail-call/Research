from collections.abc import Callable
from typing import TypeAlias

import numpy as np


Real: TypeAlias = float
Theta: TypeAlias = np.ndarray
ObjectiveFunction: TypeAlias = Callable[[Theta], Real]


Counter = 0
#MAX_COUNTER = 3500
MAX_COUNTER = 2694

def is_converged(theta_t: Theta) -> bool:
    global Counter

    if Counter > MAX_COUNTER:
        return True 
    else:
        Counter += 1
        return False


def gradient(
    func: ObjectiveFunction,
    theta_t: Theta
) -> Theta:
    dx = 0.001

    return (
        func(theta_t + dx) - func(theta_t)
    ) / (theta_t + dx - theta_t)


def adam(
    objective_function: ObjectiveFunction,
    theta_0: Theta,
    alpha: Real = 0.001,
    beta_1: Real = 0.9,
    beta_2: Real = 0.999,
    step_size: Real = 1e-8,
) -> Theta:
    m_t: np.ndarray = np.ndarray(theta_0.shape)
    v_t: np.ndarray = np.ndarray(theta_0.shape)
    t = 0

    theta_t = theta_0

    while not is_converged(theta_t):
        t = t + 1

        g_t = gradient(objective_function, theta_t)

        m_t = beta_1 * m_t + (1 - beta_1) * g_t
        v_t = beta_2 * v_t + (1 - beta_2) * (g_t ** 2)

        m_hat_t = m_t / (1 - beta_1 ** t)
        v_hat_t = v_t / (1 - beta_2 ** t)
        
        theta_t = theta_t - (
            alpha * m_hat_t / (np.sqrt(v_hat_t) + step_size)
        )
        
        print(f"theta_t = {theta_t}, g_t = {g_t} t = {t} m_t = {m_t}")

    return theta_t


def objective_function(theta: Theta) -> Real:
    return theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2


adam(
    objective_function=objective_function,
    theta_0=np.random.rand(3),
)