# Knowledge-Informed Multifidelity Modeling with Physics Toggle

This repository contains four benchmark cases for knowledge-informed multifidelity neural discrepancy learning. The code keeps the original modeling workflow from the uploaded scripts, but organizes it as a GitHub-ready project and adds a physics-loss switch so each method can be tested **with** or **without** physics loss.

The main idea is to compare whether the ansatz-guided discrepancy representation can work with data loss alone, rather than requiring a physics residual every time.

## Methods

The project includes:

1. Standard NN discrepancy correction
2. Ansatz-informed NN
3. Bayesian ansatz-informed NN with uncertainty estimates

Each method supports:

- `--physics true` or `--physics on`: include the physics residual/loss
- `--physics false` or `--physics off`: use data loss only
- `--compare-physics`: run both modes automatically

## Benchmark cases

| Case | Description |
|---|---|
| 1 | Linear Scaling |
| 2 | Moderate Nonlinearities |
| 3 | Complex Fidelity |
| 4 | Divergent Dynamics |

## Project structure

```text
ki-multifidelity-physics-toggle/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── ki_multifidelity/
│       ├── __init__.py
│       ├── cases.py
│       ├── models.py
│       ├── physics.py
│       ├── plotting.py
│       └── run.py
├── original_scripts/
│   ├── case1_original.py
│   ├── case2_original.py
│   ├── case3_original.py
│   └── case4_original.py
├── figures/
└── results/
```

## Installation

Create a new environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## Run examples

Run all four cases with physics loss enabled:

```bash
python main.py --physics true
```

Run all four cases without physics loss:

```bash
python main.py --physics false
```

Run both with and without physics loss for direct comparison:

```bash
python main.py --compare-physics
```

Run only the ansatz method without physics loss:

```bash
python main.py --methods ansatz --physics false
```

Run only Case 1 and Case 2 with fewer epochs for a quick smoke test:

```bash
python main.py --cases 1 2 --methods standard ansatz --n-hf 5 --epochs 100 --physics false
```

Run without generating plots:

```bash
python main.py --no-plots
```

## Spyder usage

In Spyder, open `main.py` and run it normally. To pass arguments in the console, use:

```python
%runfile C:/path/to/ki-multifidelity-physics-toggle/main.py --args --compare-physics
```

or:

```python
%runfile C:/path/to/ki-multifidelity-physics-toggle/main.py --args --physics false --methods ansatz
```

## Outputs

Generated outputs are saved to:

- `figures/`: prediction and discrepancy plots
- `results/summary_results.csv`: RMSE, cost, method, case, and physics mode

## Notes

The physics residual is the same form used in the original scripts:

```text
y'' + 400 y - 400 t
```

When physics is disabled, the deterministic models optimize only the HF data mismatch. For the Bayesian ansatz model, the physics likelihood term is removed and inference is driven by the data likelihood and priors.

The original uploaded scripts are kept in `original_scripts/` for traceability.
