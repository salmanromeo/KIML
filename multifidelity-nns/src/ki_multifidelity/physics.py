import tensorflow as tf


def physics_residual_tf(t, y_func):
    """Original residual used in the uploaded scripts: y'' + 400 y - 400 t."""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape() as tape1:
            tape1.watch(t)
            y_val = y_func(t)
        dy_dt = tape1.gradient(y_val, t)
    d2y_dt2 = tape2.gradient(dy_dt, t)
    del tape2
    return d2y_dt2 + 400.0 * y_val - 400.0 * t
