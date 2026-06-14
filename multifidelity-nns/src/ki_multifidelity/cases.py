from dataclasses import dataclass
from typing import Callable
import numpy as np
import tensorflow as tf

ArrayFn = Callable[[object], object]


@dataclass(frozen=True)
class Case:
    case_id: int
    title: str
    slug: str
    y_hf: ArrayFn
    y_lf: ArrayFn


def _is_tf(t) -> bool:
    return isinstance(t, tf.Tensor)


def case1_hf(t):
    return tf.sin(20.0 * t) + t if _is_tf(t) else np.sin(20 * t) + t


def case1_lf(t):
    return 0.8 * tf.sin(20.0 * t) + 0.5 * t + 0.5 if _is_tf(t) else 0.8 * np.sin(20 * t) + 0.5 * t + 0.5


def case2_hf(t):
    if _is_tf(t):
        return tf.sin(20.0 * t) + t + 0.2 * tf.sin(20.0 * t) * tf.cos(10.0 * t)
    return np.sin(20 * t) + t + 0.2 * np.sin(20 * t) * np.cos(10 * t)


def case2_lf(t):
    if _is_tf(t):
        return 0.8 * tf.sin(18.0 * t) + 0.6 * t + 0.4 + 0.15 * tf.sin(18.0 * t) * tf.sin(8.0 * t) + 0.1 * t * tf.cos(12.0 * t)
    return 0.8 * np.sin(18 * t) + 0.6 * t + 0.4 + 0.15 * np.sin(18 * t) * np.sin(8 * t) + 0.1 * t * np.cos(12 * t)


def case3_hf(t):
    if _is_tf(t):
        return tf.sin(20.0 * t) + t + 0.5 * tf.sin(5.0 * t) * tf.exp(-2.0 * t)
    return np.sin(20 * t) + t + 0.5 * np.sin(5 * t) * np.exp(-2 * t)


def case3_lf(t):
    if _is_tf(t):
        return 0.8 * tf.sin(18.0 * t + 0.5) + 0.6 * t + 0.3 + 0.4 * tf.cos(7.0 * t) * tf.exp(-1.8 * t) * (1.0 + 0.3 * tf.sin(12.0 * t))
    return 0.8 * np.sin(18 * t + 0.5) + 0.6 * t + 0.3 + 0.4 * np.cos(7 * t) * np.exp(-1.8 * t) * (1.0 + 0.3 * np.sin(12 * t))


def case4_hf(t):
    return tf.sin(20.0 * t) + t + 0.3 * t**2 - 0.2 * t**3 if _is_tf(t) else np.sin(20 * t) + t + 0.3 * t**2 - 0.2 * t**3


def case4_lf(t):
    if _is_tf(t):
        return 0.9 * tf.sin(19.0 * t) + 0.7 * t + 0.4 * tf.exp(-2.0 * t) * tf.sin(15.0 * t) + 0.2 * tf.cos(8.0 * t) / (1.0 + 2.0 * t)
    return 0.9 * np.sin(19 * t) + 0.7 * t + 0.4 * np.exp(-2 * t) * np.sin(15 * t) + 0.2 * np.cos(8 * t) / (1 + 2 * t)


CASES = {
    1: Case(1, "Case 1: Linear Scaling", "case1_linear_scaling", case1_hf, case1_lf),
    2: Case(2, "Case 2: Moderate Nonlinearities", "case2_moderate_nonlinearities", case2_hf, case2_lf),
    3: Case(3, "Case 3: Complex Fidelity", "case3_complex_fidelity", case3_hf, case3_lf),
    4: Case(4, "Case 4: Divergent Dynamics", "case4_divergent_dynamics", case4_hf, case4_lf),
}
