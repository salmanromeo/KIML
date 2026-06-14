"""Core multi-fidelity Gaussian Process routines.

The numerical workflow follows the original scripts:
1. Fit a GP to low-fidelity data.
2. Estimate autoregressive scaling rho with linear regression.
3. Fit a GP to the discrepancy for the standard co-kriging model.
4. Fit an augmented joint GP using time and fidelity level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Product, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def _case_output_dir(output_dir: str | Path, case_slug: str) -> Path:
    path = Path(output_dir) / case_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_1d(values) -> np.ndarray:
    """Return model outputs as a 1D array for sklearn and plotting consistency."""
    return np.asarray(values, dtype=float).reshape(-1)


def standard_gp_co_kriging(
    y_hf: Callable,
    y_lf: Callable,
    case_title: str,
    case_slug: str,
    n_hf: int,
    n_lf: int,
    plot_results: bool = True,
    output_dir: str | Path = "figures",
) -> Dict[str, float | int | str]:
    """Standard GP with autoregressive co-kriging: y_HF(t) = rho * y_LF(t) + delta(t)."""
    t_train_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    y_train_lf = _as_1d(y_lf(t_train_lf))
    t_train_hf = np.linspace(0, 1, n_hf).reshape(-1, 1)
    y_train_hf = _as_1d(y_hf(t_train_hf))

    kernel_lf = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)
    gp_lf = GaussianProcessRegressor(kernel=kernel_lf, n_restarts_optimizer=5, random_state=42)
    gp_lf.fit(t_train_lf, y_train_lf)

    lr = LinearRegression()
    lr.fit(_as_1d(y_lf(t_train_hf)).reshape(-1, 1), y_train_hf)
    rho = float(np.ravel(lr.coef_)[0])
    delta_train = y_train_hf - rho * _as_1d(y_lf(t_train_hf))

    kernel_delta = ConstantKernel(1.0) * RBF(length_scale=0.05) + WhiteKernel(noise_level=0.001)
    gp_delta = GaussianProcessRegressor(kernel=kernel_delta, n_restarts_optimizer=5, random_state=42)
    gp_delta.fit(t_train_hf, delta_train)

    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_lf_pred, y_lf_std = gp_lf.predict(t_test, return_std=True)
    y_lf_pred = _as_1d(y_lf_pred)
    delta_pred, delta_std = gp_delta.predict(t_test, return_std=True)
    delta_pred = _as_1d(delta_pred)
    y_hf_pred = rho * y_lf_pred + delta_pred
    y_hf_std = np.sqrt((rho * y_lf_std) ** 2 + delta_std**2)

    y_true = _as_1d(y_hf(t_test))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hf_pred)))
    complexity = 1000 * n_hf + 1 * n_lf
    true_delta = _as_1d(y_hf(t_test)) - rho * _as_1d(y_lf(t_test))

    if plot_results:
        case_dir = _case_output_dir(output_dir, case_slug)
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color="#1f77b4", linewidth=2.5, label="True HF", zorder=3)
        ax1.plot(t_test, y_hf_pred, color="#d62728", linewidth=1.8, linestyle="--", label="Standard GP Prediction", zorder=4)
        ax1.fill_between(
            t_test.flatten(),
            y_hf_pred.flatten() - 2 * y_hf_std,
            y_hf_pred.flatten() + 2 * y_hf_std,
            color="#d62728",
            alpha=0.25,
            label="95% CI",
            zorder=2,
        )
        ax1.scatter(t_train_hf, y_train_hf, s=60, color="#d62728", marker="D", label=f"HF Data ($N_{{HF}}$={n_hf})", zorder=5, edgecolors="white", linewidth=0.8)
        ax1.scatter(t_train_lf, y_train_lf, s=8, color="#2ca02c", marker="o", alpha=0.6, label=f"LF Data ($N_{{LF}}$={n_lf})", zorder=1)
        ax1.set_ylabel("Quantity of Interest, $y$", fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc="best", mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")
        ax1.set_title(f"{case_title} | Standard GP | $N_{{HF}}$={n_hf} | RMSE: {rmse:.4f} | Complexity: {complexity}", fontsize=11, pad=10)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_test, true_delta, color="#1f77b4", linewidth=2.5, label="True $\\delta_{true}(t)$", alpha=0.8)
        ax2.plot(t_test, delta_pred, color="#d62728", linewidth=2, linestyle="--", label="Learned $\\delta_{pred}(t)$")
        ax2.fill_between(t_test.flatten(), delta_pred.flatten() - 2 * delta_std, delta_pred.flatten() + 2 * delta_std, color="#d62728", alpha=0.3, label="95% CI")
        ax2.set_xlabel("Input, $t$", fontsize=11)
        ax2.set_ylabel("Discrepancy, $\\delta$", fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc="upper left", fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")
        ax2.set_title("Standard GP Discrepancy Learning: True vs Predicted", fontsize=11, pad=10)

        plt.savefig(case_dir / f"standard_gp_nhf{n_hf}.png", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return {"rmse": rmse, "complexity": complexity, "rho": rho, "N_HF": n_hf, "N_LF": n_lf, "method": "Standard GP", "case": case_title}


def joint_gp_co_kriging(
    y_hf: Callable,
    y_lf: Callable,
    case_title: str,
    case_slug: str,
    n_hf: int,
    n_lf: int,
    plot_results: bool = True,
    output_dir: str | Path = "figures",
) -> Dict[str, float | int | str]:
    """Joint GP with augmented input [t, fidelity]."""
    t_train_lf = np.linspace(0, 1, n_lf).reshape(-1, 1)
    y_train_lf = _as_1d(y_lf(t_train_lf))
    t_train_hf = np.linspace(0, 1, n_hf).reshape(-1, 1)
    y_train_hf = _as_1d(y_hf(t_train_hf))

    x_train = np.vstack([
        np.hstack([t_train_lf, np.zeros((n_lf, 1))]),
        np.hstack([t_train_hf, np.ones((n_hf, 1))]),
    ])
    y_train = np.concatenate([y_train_lf, y_train_hf])

    kernel_t = RBF(length_scale=0.1) + RBF(length_scale=0.02)
    kernel_fidelity = ConstantKernel(1.0) * RBF(length_scale=1.0)
    kernel = Product(kernel_t, kernel_fidelity) + WhiteKernel(noise_level=0.01)

    gp_mf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, random_state=42)
    gp_mf.fit(x_train, y_train)

    t_test = np.linspace(0, 1, 500).reshape(-1, 1)
    x_test = np.hstack([t_test, np.ones((500, 1))])
    y_pred, y_std = gp_mf.predict(x_test, return_std=True)

    y_true = _as_1d(y_hf(t_test))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred.flatten())))
    complexity = 1000 * n_hf + 1 * n_lf

    lr = LinearRegression()
    lr.fit(_as_1d(y_lf(t_train_hf)).reshape(-1, 1), y_train_hf)
    rho_approx = float(np.ravel(lr.coef_)[0])
    effective_delta = y_pred.flatten() - rho_approx * _as_1d(y_lf(t_test))
    true_delta_approx = _as_1d(y_hf(t_test)) - rho_approx * _as_1d(y_lf(t_test))

    if plot_results:
        case_dir = _case_output_dir(output_dir, case_slug)
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_test, y_true, color="#1f77b4", linewidth=2.5, label="True HF", zorder=3)
        ax1.plot(t_test, y_pred, color="#ff7f0e", linewidth=1.8, linestyle="--", label="Joint GP Prediction", zorder=4)
        ax1.fill_between(t_test.flatten(), y_pred.flatten() - 2 * y_std, y_pred.flatten() + 2 * y_std, color="#ff7f0e", alpha=0.25, label="95% CI", zorder=2)
        ax1.scatter(t_train_hf, y_train_hf, s=60, color="#d62728", marker="D", label=f"HF Data ($N_{{HF}}$={n_hf})", zorder=5, edgecolors="white", linewidth=0.8)
        ax1.scatter(t_train_lf, y_train_lf, s=8, color="#2ca02c", marker="o", alpha=0.6, label=f"LF Data ($N_{{LF}}$={n_lf})", zorder=1)
        ax1.set_ylabel("Quantity of Interest, $y$", fontsize=11)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-3, 3)
        ax1.legend(loc="best", mode="expand", ncol=2, fontsize=9)
        ax1.text(0.02, 0.99, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold", va="top")
        ax1.set_title(f"{case_title} | Joint GP | $N_{{HF}}$={n_hf} | RMSE: {rmse:.4f} | Complexity: {complexity}", fontsize=11, pad=10)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(t_test, true_delta_approx, color="#1f77b4", linewidth=2.5, label="True $\\delta_{true}(t)$", alpha=0.8)
        ax2.plot(t_test, effective_delta, color="#ff7f0e", linewidth=2, label="Learned $\\delta_{pred}(t)$", linestyle="--")
        ax2.fill_between(t_test.flatten(), effective_delta - 2 * y_std, effective_delta + 2 * y_std, color="#ff7f0e", alpha=0.3, label="95% CI")
        ax2.set_xlabel("Input, $t$", fontsize=11)
        ax2.set_ylabel("Discrepancy, $\\delta$", fontsize=11)
        ax2.legend(bbox_to_anchor=(0.0, 0.95), loc="upper left", fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.text(0.02, 0.99, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold", va="top")
        ax2.set_title("Joint GP Discrepancy Learning: True vs Predicted", fontsize=11, pad=10)

        plt.savefig(case_dir / f"joint_gp_nhf{n_hf}.png", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return {"rmse": rmse, "complexity": complexity, "N_HF": n_hf, "N_LF": n_lf, "method": "Joint GP", "case": case_title}
