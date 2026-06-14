# Knowledge-Informed Multifidelity Modeling

A research implementation of knowledge-informed multifidelity learning using ansatz-guided discrepancy representations. This repository compares classical multifidelity surrogate modeling approaches with knowledge-informed neural architectures across multiple benchmark cases of increasing complexity.

## Overview

Multifidelity modeling seeks to combine:

* **Low-fidelity (LF)** data: inexpensive but biased
* **High-fidelity (HF)** data: accurate but computationally expensive

Traditional discrepancy-learning approaches often struggle when the LF-HF relationship becomes nonlinear or structurally complex, especially when only a small number of HF samples are available.

This repository implements a knowledge-informed framework where prior structural knowledge is embedded through an **analytical ansatz-guided discrepancy representation**, allowing the model to learn complex LF-HF relationships efficiently.

A key finding of our work is that:

> Explicit physics residual losses are not always necessary. Ansätze-informed discrepancy representations combined with data-driven learning can achieve highly accurate multifidelity predictions while reducing model complexity and training overhead.

To investigate this claim, the repository supports both:

* **Physics-informed training** (`physics=True`)
* **Data-only training** (`physics=False`)

allowing direct comparison of the impact of physics residual losses.

---

## Methods Implemented

### Gaussian Process Models

#### Standard GP

Classical autoregressive co-Kriging formulation

$y_{HF}(t)=\rho y_{LF}(t)+\delta(t)$

where the discrepancy is modeled using a Gaussian Process.

#### Joint GP

Multi-fidelity Gaussian Process using an augmented fidelity input space.

---

### Neural Network Models

#### Standard NN

Purely data-driven discrepancy learning

$y_{HF}(t)=y_{LF}(t)+\delta_{NN}(t)$

#### Ansatz-Informed NN

Knowledge-informed discrepancy representation

$y_{HF}(t)=y_{LF}(t)+\psi(t)+\delta_{NN}(t)$

where

$\psi(t)=A\sin(\omega t+\phi)+Bt+C$

captures dominant discrepancy structures.

#### Bayesian Ansatz-Informed NN

Bayesian extension of the ansatz-informed model providing predictive uncertainty quantification.

---

## Physics Loss Toggle

This repository supports training with or without physics residual constraints.

### With Physics Loss

```bash
python main.py --physics true
```

Loss function:

```text
Total Loss = Data Loss + λ × Physics Loss
```

### Without Physics Loss

```bash
python main.py --physics false
```

Loss function:

```text
Total Loss = Data Loss
```

This feature enables direct investigation of one of the central questions addressed in our work:

> Does multifidelity discrepancy learning truly require physics residual constraints, or can structured discrepancy representations alone provide sufficient inductive bias?

---

## Benchmark Cases

The repository contains four benchmark problems of increasing difficulty:

| Case   | Description                    |
| ------ | ------------------------------ |
| Case 1 | Linear scaling relationship    |
| Case 2 | Moderate nonlinear discrepancy |
| Case 3 | Complex fidelity relationship  |
| Case 4 | Divergent fidelity dynamics    |

Each benchmark contains unique LF and HF relationships designed to stress-test multifidelity learning methods.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/salmanromeo/KIML.git

cd knowledge-informed-multifidelity
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Run all cases:

```bash
python main.py
```

Run a specific case:

```bash
python main.py --case 1
```

Run without physics constraints:

```bash
python main.py --physics false
```

Run with physics constraints:

```bash
python main.py --physics true
```

Disable plotting:

```bash
python main.py --no-plots
```

---

## Repository Structure

```text
...../
│
├── main.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── models/
│   ├── cases/
│   ├── training/
│   ├── plotting/
│   └── utils/
│
├── results/
├── figures/
└── tests/
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{romeo2026knowledge,
  title={A knowledge-informed machine learning framework for multifidelity modeling},
  author={Romeo, Shafi Al Salman and Kara, Kursat and San, Omer},
  journal={Machine Learning for Computational Science and Engineering},
  volume={2},
  number={1},
  pages={26},
  year={2026},
  publisher={Springer}
}
```

---

## Reference

The implementation accompanies the following publication:

Romeo, S. A. S., Kara, K., & San, O. (2026).
*A Knowledge-Informed Machine Learning Framework for Multifidelity Modeling.*
Machine Learning for Computational Science and Engineering, 2(1), 26.

---

## License

This repository is released for academic and research purposes. Please cite the associated publication when using this code.
