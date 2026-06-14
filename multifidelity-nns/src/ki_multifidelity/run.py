from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from .cases import CASES
from .models import standard_nn, ansatz_nn, bayesian_ansatz_nn
from .plotting import configure_plots, ensure_dir


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if normalized in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, 1/0, or on/off.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge-informed multifidelity NN benchmarks with optional physics loss.")
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4], choices=sorted(CASES), help="Case IDs to run.")
    parser.add_argument("--methods", nargs="+", default=["standard", "ansatz", "bayesian"], choices=["standard", "ansatz", "bayesian"], help="Methods to run.")
    parser.add_argument("--n-hf", type=int, nargs="+", default=[5, 10, 15], help="HF sample counts.")
    parser.add_argument("--n-lf", type=int, default=1000, help="LF sample count.")
    parser.add_argument("--physics", type=str_to_bool, default=True, help="Turn physics loss on/off: true/false, on/off.")
    parser.add_argument("--compare-physics", action="store_true", help="Run each selected method twice: with and without physics loss.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs for deterministic NN methods.")
    parser.add_argument("--physics-weight", type=float, default=0.5, help="Physics-loss weight.")
    parser.add_argument("--physics-std", type=float, default=10.0, help="Physics residual standard deviation for deterministic losses.")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Noise standard deviation for Bayesian training data.")
    parser.add_argument("--num-results", type=int, default=80, help="HMC posterior samples per chain.")
    parser.add_argument("--num-burnin-steps", type=int, default=20, help="HMC burn-in steps.")
    parser.add_argument("--n-chains", type=int, default=1, help="Number of HMC chains.")
    parser.add_argument("--no-plots", action="store_true", help="Disable figure generation.")
    parser.add_argument("--figures-dir", default="figures", help="Directory for figures.")
    parser.add_argument("--results-dir", default="results", help="Directory for CSV results.")
    parser.add_argument("--quiet", action="store_true", help="Reduce training output.")
    return parser.parse_args(argv)


def run_case(case_id: int, methods: Iterable[str], n_hf_values: Iterable[int], n_lf: int,
             physics_values: Iterable[bool], args: argparse.Namespace) -> list[dict]:
    case = CASES[case_id]
    print(f"\n{case.title}")
    print("=" * 70)
    results: list[dict] = []

    for use_physics in physics_values:
        mode = "WITH physics loss" if use_physics else "WITHOUT physics loss, data loss only"
        print(f"\nMode: {mode}")

        for method in methods:
            print(f"\nRunning {method.replace('_', ' ').title()}...")
            for n_hf in n_hf_values:
                print(f"  N_HF = {n_hf}...")
                common = dict(
                    y_hf=case.y_hf,
                    y_lf=case.y_lf,
                    case_title=case.title,
                    case_slug=case.slug,
                    n_hf=n_hf,
                    n_lf=n_lf,
                    use_physics=use_physics,
                    physics_weight=args.physics_weight,
                    plot_results=not args.no_plots,
                    output_dir=args.figures_dir,
                    verbose=not args.quiet,
                )
                if method == "standard":
                    result = standard_nn(**common, physics_std=args.physics_std, epochs=args.epochs)
                elif method == "ansatz":
                    result = ansatz_nn(**common, physics_std=args.physics_std, epochs=args.epochs)
                elif method == "bayesian":
                    result = bayesian_ansatz_nn(
                        **common,
                        noise_std=args.noise_std,
                        physics_std=8.0,
                        n_chains=args.n_chains,
                        num_results=args.num_results,
                        num_burnin_steps=args.num_burnin_steps,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                results.append(result)
                print(f"    RMSE: {result['rmse']:.4f}, Complexity: {result['complexity']}")
    return results


def write_results(results: list[dict], results_dir: str | Path) -> Path:
    results_dir = ensure_dir(results_dir)
    output_path = results_dir / "summary_results.csv"
    if not results:
        return output_path
    fieldnames = sorted({key for row in results for key in row.keys()})
    preferred = ["case", "method", "physics", "N_HF", "N_LF", "rmse", "complexity"]
    fieldnames = [x for x in preferred if x in fieldnames] + [x for x in fieldnames if x not in preferred]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return output_path


def plot_rmse_summary(results: list[dict], figures_dir: str | Path) -> None:
    if not results:
        return
    figures_dir = ensure_dir(figures_dir)
    grouped = {}
    for row in results:
        key = (row["case"], row["method"], row["physics"])
        grouped.setdefault(key, []).append(row)

    for case_title in sorted({r["case"] for r in results}):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for (case, method, physics), rows in grouped.items():
            if case != case_title:
                continue
            rows = sorted(rows, key=lambda x: x["N_HF"])
            label = f"{method} ({'physics' if physics else 'data only'})"
            ax.plot([r["N_HF"] for r in rows], [r["rmse"] for r in rows], marker="o", linewidth=2.0, label=label)
        ax.set_xlabel("Number of HF Samples", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(f"{case_title} | RMSE Comparison", fontsize=13, pad=15)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        fig.tight_layout()
        safe = case_title.lower().replace(":", "").replace(" ", "_")
        fig.savefig(figures_dir / f"{safe}_rmse_summary.png", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Case':<34} {'Method':<22} {'Physics':<8} {'N_HF':<6} {'RMSE':<12} {'Complexity':<12}")
    print("-" * 90)
    for row in results:
        print(f"{row['case']:<34} {row['method']:<22} {str(row['physics']):<8} {row['N_HF']:<6} {row['rmse']:<12.5f} {row['complexity']:<12}")


def main(argv: list[str] | None = None) -> None:
    configure_plots()
    args = parse_args(argv)
    physics_values = [True, False] if args.compare_physics else [args.physics]
    all_results: list[dict] = []
    for case_id in args.cases:
        all_results.extend(run_case(case_id, args.methods, args.n_hf, args.n_lf, physics_values, args))
    output_csv = write_results(all_results, args.results_dir)
    plot_rmse_summary(all_results, args.figures_dir)
    print_summary(all_results)
    print(f"\nSaved results to: {output_csv}")


if __name__ == "__main__":
    main()
