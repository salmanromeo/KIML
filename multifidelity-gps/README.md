# Multi-Fidelity Gaussian Process Case Studies

This repository contains multi-fidelity Gaussian Process experiments comparing two methods:

1. **Standard GP co-kriging** using the autoregressive form
$y_{HF}(t) = \rho y_{LF}(t) + \delta(t)$
2. **Joint GP co-kriging** using an augmented input space with fidelity level.

The project combines four cases with different high-fidelity and low-fidelity functions.

## Cases

| Case | Description | Source idea |
|---|---|---|
| 1 | Linear Scaling | Baseline HF/LF relationship |
| 2 | Moderate Nonlinearities | Multiplicative and nonlinear LF terms |
| 3 | Complex Fidelity | Exponential, trigonometric, and shifted LF dynamics |
| 4 | Divergent Dynamics | Polynomial HF and exponential/trigonometric LF dynamics |

## Project Structure

```text
multifidelity-gps/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
├── src/
│   └── multifidelity_gp/
│       ├── __init__.py
│       ├── cases.py
│       ├── models.py
│       ├── plotting.py
│       └── run.py
├── figures/
└── results/
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For editable package usage:

```bash
pip install -e .
```

## Usage

Run all cases:

```bash
python main.py
```

Run selected cases only:

```bash
python main.py --cases 1 2
```

Change HF sample sizes:

```bash
python main.py --n-hf 5 10 15 20
```

Run without saving plots:

```bash
python main.py --no-plots
```

## Outputs

Generated figures are saved under:

```text
figures/<case_name>/
```

The summary table is saved to:

```text
results/summary_results.csv
```

## Notes

The HF and LF equations are stored separately in `src/multifidelity_gp/cases.py`.
