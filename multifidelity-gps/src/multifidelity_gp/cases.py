"""HF/LF function definitions for all cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FidelityCase:
    """Container for one multi-fidelity test case."""

    id: int
    slug: str
    title: str
    y_hf: Callable
    y_lf: Callable


def case1_y_hf(t):
    return np.sin(20 * t) + t


def case1_y_lf(t):
    return 0.8 * np.sin(20 * t) + 0.5 * t + 0.5


def case2_y_hf(t):
    return np.sin(20 * t) + t + 0.2 * np.sin(20 * t) * np.cos(10 * t)


def case2_y_lf(t):
    return (
        0.8 * np.sin(18 * t)
        + 0.6 * t
        + 0.4
        + 0.15 * np.sin(18 * t) * np.sin(8 * t)
        + 0.1 * t * np.cos(12 * t)
    )


def case3_y_hf(t):
    return np.sin(20 * t) + t + 0.5 * np.sin(5 * t) * np.exp(-2 * t)


def case3_y_lf(t):
    return (
        0.8 * np.sin(18 * t + 0.5)
        + 0.6 * t
        + 0.3
        + 0.4 * np.cos(7 * t) * np.exp(-1.8 * t) * (1 + 0.3 * np.sin(12 * t))
    )


def case4_y_hf(t):
    return np.sin(20 * t) + t + 0.3 * t**2 - 0.2 * t**3


def case4_y_lf(t):
    return (
        0.9 * np.sin(19 * t)
        + 0.7 * t
        + 0.4 * np.exp(-2 * t) * np.sin(15 * t)
        + 0.2 * np.cos(8 * t) / (1 + 2 * t)
    )


CASES = {
    1: FidelityCase(1, "case1_linear_scaling", "Case 1: Linear Scaling", case1_y_hf, case1_y_lf),
    2: FidelityCase(2, "case2_moderate_nonlinearities", "Case 2: Moderate Nonlinearities", case2_y_hf, case2_y_lf),
    3: FidelityCase(3, "case3_complex_fidelity", "Case 3: Complex Fidelity", case3_y_hf, case3_y_lf),
    4: FidelityCase(4, "case4_divergent_dynamics", "Case 4: Divergent Dynamics", case4_y_hf, case4_y_lf),
}
