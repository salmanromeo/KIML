from pathlib import Path
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from .physics import physics_residual_tf
from .plotting import ensure_dir

tfd = tfp.distributions
tfb = tfp.bijectors


def _rho(y_lf, y_hf, t_train_hf, y_train_hf) -> float:
    lr = LinearRegression()
    lr.fit(np.asarray(y_lf(t_train_hf)).reshape(-1, 1), np.asarray(y_train_hf).reshape(-1))
    return float(np.ravel(lr.coef_)[0])


def _physics_label(use_physics: bool) -> str:
    return "with_physics" if use_physics else "without_physics"


def standard_nn(y_hf, y_lf, case_title, case_slug, n_hf, n_lf, use_physics=True,
                physics_weight=0.5, physics_std=10.0, epochs=2000, plot_results=True,
                output_dir="figures", verbose=True):
    t_train_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    y_train_lf = y_lf(t_train_lf)
    t_train_hf = np.linspace(0, 1, n_hf).reshape(-1, 1)
    y_train_hf = y_hf(t_train_hf)

    t_train_hf_tf = tf.constant(t_train_hf, dtype=tf.float32)
    y_train_hf_tf = tf.constant(y_train_hf, dtype=tf.float32)
    t_physics = tf.constant(np.linspace(0, 1, 500).reshape(-1, 1), dtype=tf.float32)

    model = keras.Sequential([
        keras.layers.Input(shape=(1,)),
        keras.layers.Dense(50, activation="tanh"),
        keras.layers.Dense(50, activation="tanh"),
        keras.layers.Dense(50, activation="tanh"),
        keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            delta_pred = model(t_train_hf_tf)
            y_pred_hf = y_lf(t_train_hf_tf) + delta_pred
            loss_data = tf.reduce_mean(tf.square(y_pred_hf - y_train_hf_tf))

            loss_physics = tf.constant(0.0, dtype=tf.float32)
            if use_physics:
                def total_solution(t):
                    return y_lf(t) + model(t)
                physics_res = physics_residual_tf(t_physics, total_solution)
                loss_physics = tf.reduce_mean(tf.square(physics_res)) / (physics_std ** 2)

            total_loss = loss_data + physics_weight * loss_physics if use_physics else loss_data

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss, loss_data, loss_physics

    if verbose:
        mode = "WITH physics loss" if use_physics else "WITHOUT physics loss"
        print(f"Training Standard NN with {n_hf} HF samples ({mode})...")
    for epoch in range(epochs):
        total_loss, loss_d, loss_p = train_step()
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}, Data Loss = {loss_d:.4f}, Physics Loss = {loss_p:.4f}")

    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_true = y_hf(t_test)
    delta_pred = model(tf.constant(t_test, dtype=tf.float32)).numpy()
    y_pred = y_lf(t_test) + delta_pred
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    complexity = 1000 * n_hf + n_lf

    rho = _rho(y_lf, y_hf, t_train_hf, y_train_hf)
    true_delta = y_hf(t_test) - rho * y_lf(t_test)

    if plot_results:
        _plot_deterministic(case_title, case_slug, "Standard NN", "#9467bd", t_test, y_true, y_pred,
                            t_train_hf, y_train_hf, t_train_lf, y_train_lf, true_delta, delta_pred,
                            n_hf, n_lf, rmse, complexity, use_physics, output_dir)

    return {"case": case_title, "rmse": rmse, "complexity": complexity, "N_HF": n_hf, "N_LF": n_lf,
            "method": "Standard NN", "physics": use_physics}


def ansatz_nn(y_hf, y_lf, case_title, case_slug, n_hf, n_lf, use_physics=True,
              physics_weight=0.5, physics_std=10.0, epochs=2000, plot_results=True,
              output_dir="figures", verbose=True):
    t_train_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    y_train_lf = y_lf(t_train_lf)
    t_train_hf = np.linspace(0, 1, n_hf).reshape(-1, 1)
    y_train_hf = y_hf(t_train_hf)

    t_train_hf_tf = tf.constant(t_train_hf, dtype=tf.float32)
    y_train_hf_tf = tf.constant(y_train_hf, dtype=tf.float32)
    t_physics = tf.constant(np.linspace(0, 1, 500).reshape(-1, 1), dtype=tf.float32)

    A = tf.Variable(1.0, dtype=tf.float32, name="amplitude")
    omega = tf.Variable(20.0, dtype=tf.float32, name="frequency")
    phi = tf.Variable(0.0, dtype=tf.float32, name="phase")
    B = tf.Variable(1.0, dtype=tf.float32, name="slope")
    C = tf.Variable(0.0, dtype=tf.float32, name="intercept")

    nn_input = keras.layers.Input(shape=(1,))
    x = keras.layers.Dense(50, activation="tanh")(nn_input)
    x = keras.layers.Dense(50, activation="tanh")(x)
    x = keras.layers.Dense(50, activation="tanh")(x)
    nn_output = keras.layers.Dense(1)(x)
    nn_model = keras.Model(nn_input, nn_output)

    def ansatz_model(t):
        base_solution = A * tf.sin(omega * t + phi) + B * t + C
        correction = nn_model(t)
        return base_solution + correction

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            y_pred_hf = ansatz_model(t_train_hf_tf)
            loss_data = tf.reduce_mean(tf.square(y_pred_hf - y_train_hf_tf))

            loss_physics = tf.constant(0.0, dtype=tf.float32)
            if use_physics:
                physics_res = physics_residual_tf(t_physics, ansatz_model)
                loss_physics = tf.reduce_mean(tf.square(physics_res)) / (physics_std ** 2)

            total_loss = loss_data + physics_weight * loss_physics if use_physics else loss_data

        trainable_vars = [A, omega, phi, B, C] + nn_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        return total_loss, loss_data, loss_physics

    if verbose:
        mode = "WITH physics loss" if use_physics else "WITHOUT physics loss"
        print(f"Training Ansatz NN with {n_hf} HF samples ({mode})...")
    for epoch in range(epochs):
        total_loss, loss_d, loss_p = train_step()
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}, Data Loss = {loss_d:.4f}, Physics Loss = {loss_p:.4f}")

    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_true = y_hf(t_test)
    y_pred = ansatz_model(tf.constant(t_test, dtype=tf.float32)).numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    complexity = 1000 * n_hf + n_lf

    rho = _rho(y_lf, y_hf, t_train_hf, y_train_hf)
    true_delta = y_hf(t_test) - rho * y_lf(t_test)
    effective_delta = y_pred.flatten() - rho * y_lf(t_test).flatten()

    if plot_results:
        _plot_deterministic(case_title, case_slug, "Ansatz NN", "m", t_test, y_true, y_pred,
                            t_train_hf, y_train_hf, t_train_lf, y_train_lf, true_delta, effective_delta,
                            n_hf, n_lf, rmse, complexity, use_physics, output_dir)

    return {"case": case_title, "rmse": rmse, "complexity": complexity, "N_HF": n_hf, "N_LF": n_lf,
            "method": "Ansatz NN", "physics": use_physics,
            "A": float(A.numpy()), "omega": float(omega.numpy()), "phi": float(phi.numpy()),
            "B": float(B.numpy()), "C": float(C.numpy())}


def bayesian_ansatz_nn(y_hf, y_lf, case_title, case_slug, n_hf, n_lf, use_physics=True,
                       noise_std=0.1, physics_weight=0.5, physics_std=8.0, n_chains=1,
                       num_results=80, num_burnin_steps=20, plot_results=True,
                       output_dir="figures", verbose=True):
    np.random.seed(42)
    t_train_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    y_train_lf = y_lf(t_train_lf) + np.random.normal(0, noise_std, t_train_lf.shape)
    t_train_hf = np.linspace(0, 1, n_hf).reshape(-1, 1)
    y_train_hf = y_hf(t_train_hf) + np.random.normal(0, noise_std, t_train_hf.shape)
    t_physics = np.linspace(0, 1, 100).reshape(-1, 1)

    t_train_hf_tf = tf.constant(t_train_hf, dtype=tf.float32)
    y_train_hf_tf = tf.constant(y_train_hf, dtype=tf.float32)
    t_physics_tf = tf.constant(t_physics, dtype=tf.float32)

    def make_bayesian_ansatz_model():
        return tfd.JointDistributionNamed({
            "A": tfd.Normal(loc=1.0, scale=2.0),
            "omega": tfd.Normal(loc=20.0, scale=5.0),
            "phi": tfd.Normal(loc=0.0, scale=1.0),
            "B": tfd.Normal(loc=1.0, scale=1.0),
            "C": tfd.Normal(loc=0.0, scale=1.0),
            "nn_weight": tfd.Normal(loc=0.0, scale=1.0),
            "nn_bias": tfd.Normal(loc=0.0, scale=1.0),
            "noise_std": tfd.HalfNormal(scale=0.5),
        })

    def bayesian_ansatz(t, params):
        analytical_part = params["A"] * tf.sin(params["omega"] * t + params["phi"]) + params["B"] * t + params["C"]
        nn_correction = params["nn_weight"] * tf.sin(40.0 * t) + params["nn_bias"] * tf.cos(40.0 * t)
        return analytical_part + 0.1 * nn_correction

    def physics_residual_bayesian(t, params):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                y_val = bayesian_ansatz(t, params)
            dy_dt = tape1.gradient(y_val, t)
        d2y_dt2 = tape2.gradient(dy_dt, t)
        del tape2
        return d2y_dt2 + 400.0 * y_val - 400.0 * t

    def target_log_prob_fn(A, omega, phi, B, C, nn_weight, nn_bias, noise_std):
        params = {"A": A, "omega": omega, "phi": phi, "B": B, "C": C, "nn_weight": nn_weight, "nn_bias": nn_bias}
        model = make_bayesian_ansatz_model()
        log_prior = model.log_prob({**params, "noise_std": noise_std})
        y_pred = bayesian_ansatz(t_train_hf_tf, params)
        log_likelihood_data = tf.reduce_sum(tfd.Normal(loc=y_pred, scale=noise_std).log_prob(y_train_hf_tf))

        if not use_physics:
            return log_prior + log_likelihood_data

        physics_res = physics_residual_bayesian(t_physics_tf, params)
        log_likelihood_physics = tf.reduce_sum(tfd.Normal(loc=0.0, scale=physics_std).log_prob(physics_res))
        return log_prior + log_likelihood_data + physics_weight * log_likelihood_physics

    if verbose:
        mode = "WITH physics likelihood" if use_physics else "WITHOUT physics likelihood"
        print(f"Running Bayesian Ansatz NN with {n_hf} HF samples ({mode})...")

    initial_state = [
        tf.constant(1.0, dtype=tf.float32), tf.constant(20.0, dtype=tf.float32),
        tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32),
        tf.constant(0.0, dtype=tf.float32), tf.constant(0.1, dtype=tf.float32),
        tf.constant(0.1, dtype=tf.float32), tf.constant(noise_std, dtype=tf.float32),
    ]
    unconstraining_bijectors = [tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Identity(), tfb.Exp()]
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_log_prob_fn, step_size=0.005, num_leapfrog_steps=5)
    transformed_kernel = tfp.mcmc.TransformedTransitionKernel(inner_kernel=hmc_kernel, bijector=unconstraining_bijectors)

    all_samples = []
    kernel_results = None
    for chain in range(n_chains):
        if verbose:
            print(f"Chain {chain + 1}/{n_chains}")
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=transformed_kernel,
            trace_fn=lambda _, pkr: pkr,
        )
        all_samples.append(samples)

    acceptance_rate = float(tf.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, tf.float32)).numpy()) if kernel_results is not None else np.nan
    if verbose:
        print(f"Acceptance rate: {acceptance_rate:.3f}")

    combined_samples = []
    for i in range(len(all_samples[0])):
        combined_samples.append(tf.concat([samples[i] for samples in all_samples], axis=0))

    t_test = np.linspace(0, 1, 200).reshape(-1, 1)
    t_test_tf = tf.constant(t_test, dtype=tf.float32)
    rho = _rho(y_lf, y_hf, t_train_hf, y_train_hf)
    n_posterior_samples = min(200, int(combined_samples[0].shape[0]))
    posterior_predictions, delta_predictions, analytical_params = [], [], []

    for _ in range(n_posterior_samples):
        sample_idx = np.random.randint(0, combined_samples[0].shape[0])
        params = {"A": combined_samples[0][sample_idx], "omega": combined_samples[1][sample_idx],
                  "phi": combined_samples[2][sample_idx], "B": combined_samples[3][sample_idx],
                  "C": combined_samples[4][sample_idx], "nn_weight": combined_samples[5][sample_idx],
                  "nn_bias": combined_samples[6][sample_idx]}
        y_pred_sample = bayesian_ansatz(t_test_tf, params).numpy()
        posterior_predictions.append(y_pred_sample)
        delta_predictions.append(y_pred_sample - rho * y_lf(t_test))
        analytical_params.append([params["A"].numpy(), params["omega"].numpy(), params["phi"].numpy(), params["B"].numpy(), params["C"].numpy()])

    posterior_predictions = np.array(posterior_predictions)
    delta_predictions = np.array(delta_predictions)
    analytical_params = np.array(analytical_params)
    y_pred_mean = np.mean(posterior_predictions, axis=0)
    y_pred_std = np.std(posterior_predictions, axis=0)
    effective_delta_mean = np.mean(delta_predictions, axis=0)
    effective_delta_std = np.std(delta_predictions, axis=0)

    y_true = y_hf(t_test)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_mean)))
    complexity = 1000 * n_hf + n_lf
    true_delta = y_hf(t_test) - rho * y_lf(t_test)

    if plot_results:
        _plot_bayesian(case_title, case_slug, t_test, y_true, y_pred_mean, y_pred_std,
                       t_train_hf, y_train_hf, t_train_lf, y_train_lf, true_delta,
                       effective_delta_mean, effective_delta_std, n_hf, n_lf, rmse,
                       complexity, use_physics, output_dir)

    return {"case": case_title, "rmse": rmse, "complexity": complexity, "N_HF": n_hf, "N_LF": n_lf,
            "noise_std": noise_std, "method": "Bayesian Ansatz NN", "physics": use_physics,
            "acceptance_rate": acceptance_rate,
            "A_mean": float(np.mean(analytical_params[:, 0])), "omega_mean": float(np.mean(analytical_params[:, 1])),
            "phi_mean": float(np.mean(analytical_params[:, 2])), "B_mean": float(np.mean(analytical_params[:, 3])),
            "C_mean": float(np.mean(analytical_params[:, 4]))}


def _plot_deterministic(case_title, case_slug, method, color, t_test, y_true, y_pred, t_train_hf, y_train_hf,
                        t_train_lf, y_train_lf, true_delta, learned_delta, n_hf, n_lf, rmse, complexity,
                        use_physics, output_dir):
    out = ensure_dir(Path(output_dir) / case_slug / _physics_label(use_physics))
    fig = plt.figure(figsize=(7, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_test, y_true, color="#1f77b4", linewidth=2.5, label="True HF", zorder=3)
    ax1.plot(t_test, y_pred, color=color, linewidth=1.8, linestyle="--", label=f"{method} Prediction", zorder=4)
    ax1.scatter(t_train_hf, y_train_hf, s=60, color="#d62728", marker="D", label=f"HF Data ($N_{{HF}}$={n_hf})", zorder=5, edgecolors="white", linewidth=0.8)
    ax1.scatter(t_train_lf, y_train_lf, s=8, color="#2ca02c", marker="o", alpha=0.6, label=f"LF Data ($N_{{LF}}$={n_lf})", zorder=1)
    ax1.set_ylabel("Quantity of Interest, $y$", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-3, 3)
    ax1.legend(loc="best", mode="expand", ncol=2, fontsize=9)
    ax1.text(0.02, 0.99, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")
    suffix = "with physics loss" if use_physics else "data loss only"
    ax1.set_title(f"{case_title} | {method} | {suffix} | $N_{{HF}}$={n_hf} | RMSE: {rmse:.4f} | Complexity: {complexity}", fontsize=10, pad=10)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_test, true_delta, color="#1f77b4", linewidth=2.5, label="True $\\delta_{true}(t)$", alpha=0.8)
    ax2.plot(t_test, learned_delta, color=color, linewidth=2, linestyle="--", label="Learned $\\delta_{pred}(t)$")
    ax2.set_xlabel("Input, $t$", fontsize=11)
    ax2.set_ylabel("Discrepancy, $\\delta$", fontsize=11)
    ax2.legend(bbox_to_anchor=(0.0, 0.95), loc="upper left", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.text(0.02, 0.99, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")
    ax2.set_title(f"{method} Discrepancy Learning: True vs Predicted", fontsize=11, pad=10)
    plt.savefig(out / f"{method.replace(' ', '_')}_NHF{n_hf}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_bayesian(case_title, case_slug, t_test, y_true, y_pred_mean, y_pred_std, t_train_hf, y_train_hf,
                   t_train_lf, y_train_lf, true_delta, effective_delta_mean, effective_delta_std, n_hf,
                   n_lf, rmse, complexity, use_physics, output_dir):
    out = ensure_dir(Path(output_dir) / case_slug / _physics_label(use_physics))
    y_pred_upper = y_pred_mean + 2 * y_pred_std
    y_pred_lower = y_pred_mean - 2 * y_pred_std
    effective_delta_high = effective_delta_mean + 2 * effective_delta_std
    effective_delta_low = effective_delta_mean - 2 * effective_delta_std

    fig = plt.figure(figsize=(7, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_test, y_true, color="#1f77b4", linewidth=2.5, label="True HF", zorder=3)
    ax1.plot(t_test, y_pred_mean, color="#e377c2", linewidth=1.8, linestyle="--", label="Bayesian Ansatz NN", zorder=4)
    ax1.fill_between(t_test.flatten(), y_pred_lower.flatten(), y_pred_upper.flatten(), color="#e377c2", alpha=0.3, label="95% CI", zorder=2)
    ax1.scatter(t_train_hf, y_train_hf, s=60, color="#d62728", marker="D", label=f"HF Data ($N_{{HF}}$={n_hf})", zorder=5, edgecolors="white", linewidth=0.8)
    ax1.scatter(t_train_lf, y_train_lf, s=8, color="#2ca02c", marker="o", alpha=0.6, label=f"LF Data ($N_{{LF}}$={n_lf})", zorder=1)
    ax1.set_ylabel("Quantity of Interest, $y$", fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-3, 3)
    ax1.legend(loc="best", mode="expand", ncol=2, fontsize=9)
    ax1.text(0.02, 0.99, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")
    suffix = "with physics likelihood" if use_physics else "data likelihood only"
    ax1.set_title(f"{case_title} | Bayesian Ansatz NN | {suffix} | $N_{{HF}}$={n_hf} | RMSE: {rmse:.4f} | Complexity: {complexity}", fontsize=10, pad=10)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_test, true_delta, color="#1f77b4", linewidth=2.5, label="True $\\delta_{true}(t)$", alpha=0.8)
    ax2.plot(t_test, effective_delta_mean, color="#e377c2", linewidth=2, linestyle="--", label="Learned $\\delta_{pred}(t)$")
    ax2.fill_between(t_test.squeeze(-1), effective_delta_low.squeeze(-1), effective_delta_high.squeeze(-1), color="#e377c2", alpha=0.3, label="95% CI")
    ax2.set_xlabel("Input, $t$", fontsize=11)
    ax2.set_ylabel("Discrepancy, $\\delta$", fontsize=11)
    ax2.legend(bbox_to_anchor=(0.0, 0.95), loc="upper left", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.text(0.02, 0.99, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")
    ax2.set_title("Bayesian Ansatz NN Discrepancy Learning: True vs Predicted", fontsize=11, pad=10)
    plt.savefig(out / f"Bayesian_Ansatz_NN_NHF{n_hf}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
