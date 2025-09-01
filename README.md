# Linear Programming Solver

This project implements **Linear Programming (LP)** techniques for solving constrained optimization problems. It supports two solution methods:

* **Graphical Method** – for problems with exactly two variables, providing visual plots of constraints, feasible regions, and the optimal solution.
* **Simplex Method** – for problems with two or more variables, presented through tableau iterations using the Big-M method where needed.

The program provides a Streamlit-based interface to input problem data, select the solving method, and obtain results in either graphical or tabular form.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nadaYossef/Linear-Programming-Problem
cd Linear-Programming-Problem
```

Install dependencies:

```bash
pip install numpy matplotlib streamlit
```

---

## Usage

Run the application with Streamlit:

```bash
streamlit run app(1).py
```

Once launched, the interface allows you to:

* Select the solving method (**Graphical** or **Simplex**).
* Choose between **Maximization** or **Minimization** problems.
* Enter the coefficients of the objective function.
* Define the constraints and their inequality type (≤, ≥, =).
* View the solution as **plots** (for graphical method) or **tableau iterations** (for simplex method).

---

## Demonstrations

### 1. Graphical Method (2 Variables)

**Example Problem:**

$$
\text{Minimize } Z = 2x + 3y
$$

Subject to:

$$
\begin{aligned}
x + y &\geq 6 \\
2x + y &\geq 7 \\
x + 4y &\geq 8 \\
x, y &\geq 0
\end{aligned}
$$

![Graphical Method Step](https://github.com/nadaYossef/Linear-Programming-Problem/raw/main/Graphical%20Method/8.png)

* **Please Find Step By Step Solution Screenshots in The "Graphical Method" Folder**

---

### 2. Simplex Method (2+ Variables)

**Example Problem:**

$$
\text{Maximize } Z = 2x_1 + 4x_2 + x_3 + x_4
$$

Subject to:

$$
\begin{aligned}
x_1 + 3x_2 + x_4 &\leq 4 \\
2x_1 + x_2 &\leq 3 \\
x_2 + 4x_3 + x_4 &\leq 3 \\
x_1, x_2, x_3, x_4 &\geq 0
\end{aligned}
$$

![Simplex Method Step](https://github.com/nadaYossef/Linear-Programming-Problem/raw/main/Simplex%20methods/7.png)

* **Please Find Step By Step Solution Screenshots in The "Simplex Method" Folder***

---

## ✅ Features

* Supports **Maximization** and **Minimization** problems.
* Provides **graphical visualization** of feasible regions and solutions for 2D problems.
* Full **Simplex algorithm** implementation with support for slack, surplus, and artificial variables.
* **Step-by-step tableau updates** displayed for clarity and transparency.
