# Rational Learning

**Official implementation for the paper:** *"A Rational Learning Method for Aerodynamic Modeling"* by Jing Li and Changtong Luo.

## 📖 Overview

This repository implements a deterministic symbolic regression framework termed **Rational Learning**. Unlike traditional stochastic search methods based on Genetic Programming (GP) , this module strictly restricts the search space to a subset of multivariate rational functions.

The core objective of the program is to discover the optimal multivariate rational function model from a given dataset $\{\mathbf{x}^{(i)}, y^{(i)}\}$, formulated as:

$$\mathcal{R}(\mathbf{x}) = \frac{\mathcal{N}(\mathbf{x})}{\mathcal{D}(\mathbf{x})} = \frac{a_0 + \sum_{i=1}^{k} a_i \cdot \phi_i(\mathbf{x})}{1 + \sum_{j=1}^{l} b_j \cdot \psi_j(\mathbf{x})}$$

where $\phi_i(\mathbf{x})$ and $\psi_j(\mathbf{x})$ are polynomial basis functions selected from a predefined library. To ensure the uniqueness of the solution and avoid scaling ambiguity, the constant term of the denominator is strictly fixed to 1.

## ⚙️ Core Hyperparameters & Complexity Constraints

To balance model expressiveness and parsimony, and to make the search problem computationally tractable , this program explicitly controls the algorithmic structure using only **three hyperparameters** (corresponding to Section 2.1 of the paper):

- **`degree` ($S_{max}$)**: The maximum total degree (power sum) of a single monomial basis function. For example, when $S_{max}=3$, terms like $x_1 x_2^2$ are admissible, but high-order nonlinear terms lacking physical significance (e.g., $x_1^2 x_2^2$) are excluded.
- **`max_num_terms` ($K_{num}$)**: The maximum number of non-zero terms allowed in the numerator (excluding the constant term).
- **`max_den_terms` ($K_{den}$)**: The maximum number of non-zero terms allowed in the denominator (excluding the constant term).

## 🧠 Core Algorithm Logic: Two-stage Optimization

This module avoids the traditional approach of directly performing computationally expensive nonlinear optimization in a vast space. Instead, it employs an efficient hybrid optimization strategy (corresponding to Section 2.2 and Algorithm 2 of the paper):

### Stage 1: Linearized Estimation

For each candidate rational structure $(k, l)$ traversed in the space, the program first transforms the original nonlinear equation $y = \mathcal{N}/(1+\mathcal{D})$ into a linear form $\mathcal{N} - y\mathcal{D} = y$. By constructing a design matrix $\mathbf{A}$ and applying standard linear least-squares for an analytical resolution , the program rapidly obtains a high-quality initial set of coefficients $\boldsymbol{\theta}^*$ for the current structure at a very low computational cost.

### Stage 2: Nonlinear Refinement

To mitigate the weighting bias introduced by the linearization process and recover the accuracy of the original physical model , the program subsequently invokes the Levenberg-Marquardt (LM) algorithm via `LsqFit.jl`. Using $\boldsymbol{\theta}^*$ obtained in Stage 1 as the initial guess , it performs local, fine-grained optimization on the original geometric residuals. This two-stage strategy significantly reduces the risk of convergence to poor local minima that commonly arise in nonlinear optimization, ensuring the robustness of the solution.

### Pareto Frontier Construction

Instead of outputting a single model , the program performs a grid-based traversal over all admissible combinations of numerator and denominator term counts , recording the optimal solution at each complexity level. Ultimately, it outputs a complete Pareto frontier. This allows users to examine the trade-off between model complexity and fitting accuracy and to select the expression that offers the most physically interpretable description based on domain knowledge.

## ⚙️ Installation

This algorithm is implemented in **Julia**. To set up the environment and install all required dependencies, please run the following commands in your Julia REPL:

```
# Clone the repository
# git clone https://github.com/YourUsername/RationalLearning.jl.git
# cd RationalLearning.jl

# Enter the package manager and instantiate the environment
julia> ]
pkg> activate .
pkg> instantiate
```

The required standard libraries and packages include `Combinatorics`, `LsqFit`, `DataFrames`, `CSV`, `Symbolics`, `Statistics`, and `Printf`.

## 🚀 Quick Start & Reproducibility

To ensure the reproducibility of the results presented in our paper, we provide self-contained examples.

You can run the analytical benchmark case (F.3 in the paper)  to observe the algorithm's capability to exactly recover rational structures:

```
julia examples/test.jl
```

The algorithm will generate a Pareto frontier of candidate models, allowing users to examine the trade-off between model complexity and fitting accuracy. Detailed results will be saved in `results.txt` and `results.csv`.

## 📂 Project Structure

- `src/RationalLearning.jl`: The core algorithm containing the basis generation, linearization stage, and nonlinear refinement solver.
- `examples/`: Contains reproducibility scripts for the benchmark cases.

## 📝 Citation

If you find this code or our methodology useful in your research, please consider citing our paper:

```
@article{li2026rational,
  title={A Rational Learning Method for Aerodynamic Modeling},
  author={Li, Jing and Luo, Changtong},
  journal={TBD},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.



