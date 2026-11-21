# Linear Programming Solver

An interactive Streamlit app that explains two methods for solving linear optimization problems with nonnegative variables.

| Method | Inputs | Output |
| --- | --- | --- |
| Graphical | Two variables | Constraint lines, feasible region, corner points, and the best feasible point |
| Simplex | Two or more variables | Tableau iterations, pivot choices, and the final objective and variable values |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install streamlit numpy matplotlib pandas
python -m streamlit run "app (1).py"
```

In the sidebar, choose a method and whether to maximize or minimize. Enter the number of constraints, then enter the objective coefficients and each constraint’s coefficients, inequality type, and right-hand side. Select **Solve LP**. Review all coefficients after switching methods because some inputs can carry over.

## Worked examples

### Graphical minimization

Minimize `2x₁ + 3x₂` subject to `x₁ + x₂ ≥ 6`, `2x₁ + x₂ ≥ 7`, and `x₁ + 4x₂ ≥ 8`. The optimum is **x₁ = 5⅓, x₂ = ⅔**, with **Z ≈ 12.67**. The app plots the feasible corner points and highlights the optimum.

### Simplex maximization

Maximize `2x₁ + 4x₂ + x₃ + x₄` subject to these constraints.

| x₁ | x₂ | x₃ | x₄ | Type | RHS |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 1 | 3 | 0 | 1 | ≤ | 4 |
| 2 | 1 | 0 | 0 | ≤ | 3 |
| 0 | 1 | 4 | 1 | ≤ | 3 |

The optimum is **(1, 1, 0.5, 0)**, with **Z = 6.50**.

Both examples returned the same objective values in Streamlit interaction tests and an independent SciPy solve. The graphical plot uses a finite viewing window. These examples cover one bounded problem per method.
