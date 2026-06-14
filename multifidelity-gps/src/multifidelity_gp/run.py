"""Command-line runner for all multi-fidelity GP cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.multifidelity_gp.cases import CASES
from src.multifidelity_gp.models import joint_gp_co_kriging, standard_gp_co_kriging
from src.multifidelity_gp.plotting import configure_plotting


def run_case(case, n_hf_samples, n_lf_samples, plot_results=True, output_dir="figures"):
    standard_results = []
    joint_results = []

    print(f"\n{case.title}")
    print("=" * 70)

    print("\nRunning Standard GP...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        result = standard_gp_co_kriging(
            y_hf=case.y_hf,
            y_lf=case.y_lf,
            case_title=case.title,
            case_slug=case.slug,
            n_hf=n_hf,
            n_lf=n_lf_samples,
            plot_results=plot_results,
            output_dir=output_dir,
        )
        standard_results.append(result)
        print(f"    RMSE: {result['rmse']:.4f}, Complexity: {result['complexity']}")

    print("\nRunning Joint GP...")
    for n_hf in n_hf_samples:
        print(f"  N_HF = {n_hf}...")
        result = joint_gp_co_kriging(
            y_hf=case.y_hf,
            y_lf=case.y_lf,
            case_title=case.title,
            case_slug=case.slug,
            n_hf=n_hf,
            n_lf=n_lf_samples,
            plot_results=plot_results,
            output_dir=output_dir,
        )
        joint_results.append(result)
        print(f"    RMSE: {result['rmse']:.4f}, Complexity: {result['complexity']}")

    if plot_results:
        plot_case_comparison(case, n_hf_samples, standard_results, joint_results, output_dir)

    return standard_results + joint_results


def plot_case_comparison(case, n_hf_samples, standard_results, joint_results, output_dir):
    case_dir = Path(output_dir) / case.slug
    case_dir.mkdir(parents=True, exist_ok=True)

    standard_rmses = [r["rmse"] for r in standard_results]
    joint_rmses = [r["rmse"] for r in joint_results]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(n_hf_samples, standard_rmses, "o-", color="#d62728", linewidth=2.5, markersize=8, label="Standard GP", markerfacecolor="white", markeredgewidth=2)
    ax.plot(n_hf_samples, joint_rmses, "s-", color="#ff7f0e", linewidth=2.5, markersize=8, label="Joint GP", markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("Number of HF Samples", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(f"{case.title} | GP Methods Comparison: RMSE vs Data Quantity", fontsize=13, pad=15)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(case_dir / "gp_comparison_metrics.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_summary(results):
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY: STANDARD GP vs JOINT GP")
    print("=" * 90)
    print(f"\n{'Case':<35} {'Method':<20} {'N_HF':<6} {'RMSE':<10} {'Complexity':<12}")
    print("-" * 90)
    for result in results:
        print(f"{result['case']:<35} {result['method']:<20} {result['N_HF']:<6} {result['rmse']:<10.4f} {result['complexity']:<12}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-fidelity Gaussian Process case studies.")
    parser.add_argument("--cases", nargs="+", type=int, default=[1, 2, 3, 4], choices=CASES.keys(), help="Case IDs to run.")
    parser.add_argument("--n-hf", nargs="+", type=int, default=[5, 10, 15], help="High-fidelity sample sizes.")
    parser.add_argument("--n-lf", type=int, default=1000, help="Low-fidelity sample size.")
    parser.add_argument("--output-dir", default="figures", help="Directory for generated figures.")
    parser.add_argument("--results-dir", default="results", help="Directory for result tables.")
    parser.add_argument("--no-plots", action="store_true", help="Run models without saving plots.")
    args, _unknown = parser.parse_known_args()

    configure_plotting()

    all_results = []
    for case_id in args.cases:
        all_results.extend(
            run_case(
                CASES[case_id],
                n_hf_samples=args.n_hf,
                n_lf_samples=args.n_lf,
                plot_results=not args.no_plots,
                output_dir=args.output_dir,
            )
        )

    print_summary(all_results)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "summary_results.csv", index=False)
    print(f"\nSaved summary table to {results_dir / 'summary_results.csv'}")


if __name__ == "__main__":
    main()
