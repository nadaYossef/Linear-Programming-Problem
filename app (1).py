import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 

plt.rcParams['figure.dpi'] = 100  # for better image quality
plt.rcParams['savefig.dpi'] = 100 

def solve_graphical(obj_coeffs, constraints_data, constraint_types, problem_type):
    st.subheader("Graphical Method Steps")
    max_coord = max([c[-1] for c in constraints_data] + [10]) * 1.5 # Scale based on RHS values
    if max_coord == 0: # Handle cases where all RHS are 0
        max_coord = 10
    
    x = np.linspace(0, max_coord, 400)
    y = np.linspace(0, max_coord, 400)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Feasible Region and Optimal Solution")
    ax.grid(True)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.8, label='x ≥ 0')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8, label='y ≥ 0')
    ax.set_xlim(0, max_coord)
    ax.set_ylim(0, max_coord)


    # Initialize the feasible region mask with non-negativity constraints
    feasible_region_mask = (X >= -1e-9) & (Y >= -1e-9) # Use tolerance for >=0

    all_lines_info = [] 

    st.write("### Step 1: Plotting Constraints")
    st.write("For each inequality, we treat it as an equation to draw the boundary line. We then shade the area representing the valid side of the inequality. The non-negativity constraints ($x \ge 0, y \ge 0$) also define boundaries.")

    for i, constr_row in enumerate(constraints_data):
        a, b, rhs = constr_row[0], constr_row[1], constr_row[2]
        constr_type = constraint_types[i]
        label = f"C{i+1}: {a:.2f}x + {b:.2f}y {constr_type} {rhs:.2f}"
        
        # Plot the line (boundary of the constraint)
        if b != 0:
            y_line = (rhs - a * x) / b
            ax.plot(x, y_line, label=label, linestyle='-')
        elif a != 0:
            ax.axvline(rhs / a, label=label, linestyle='-')
        else:
            st.warning(f"Constraint {i+1} is trivial or ill-defined: {a}x + {b}y {constr_type} {rhs}")
            continue

        # Update the feasible region mask based on the current constraint
        if constr_type == "<=":
            feasible_region_mask &= (a * X + b * Y <= rhs + 1e-9) 
        elif constr_type == ">=":
            feasible_region_mask &= (a * X + b * Y >= rhs - 1e-9) 
        elif constr_type == "=":
            feasible_region_mask &= (np.abs(a * X + b * Y - rhs) < 1e-3) # Use small tolerance for equality
            st.write(f"Note: Equality constraint C{i+1} defines a strict line.")

        all_lines_info.append({'coeffs': [a, b], 'rhs': rhs, 'type': constr_type, 'label': label})

    # Plot the combined feasible region (lightly shaded)
    ax.imshow(feasible_region_mask.astype(float), extent=(x.min(), x.max(), y.min(), y.max()), 
              origin="lower", cmap='Greys', alpha=0.2, zorder=0)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    st.pyplot(fig)
    st.markdown("---")

    st.write("### Step 2: Identifying the Feasible Solution Region and its Corner Points")
    st.write("The **feasible region** is the area where all constraints (including $x \ge 0, y \ge 0$) are simultaneously satisfied. The optimal solution for a Linear Programming problem always occurs at one of the **corner points** of this region. We find these corner points by calculating the intersections of the constraint lines and checking if they fall within the feasible region.")

    feasible_points = []
    
    # Add axes lines for intersection calculations
    # x=0 line
    all_lines_info.append({'coeffs': [1, 0], 'rhs': 0, 'type': '>=', 'label': 'x=0', 'is_axis': True}) 
    # y=0 line
    all_lines_info.append({'coeffs': [0, 1], 'rhs': 0, 'type': '>=', 'label': 'y=0', 'is_axis': True}) 

    # Find intersection points of all pairs of lines
    for i in range(len(all_lines_info)):
        for j in range(i + 1, len(all_lines_info)):
            line1 = all_lines_info[i]
            line2 = all_lines_info[j]

            A_matrix = np.array([[line1['coeffs'][0], line1['coeffs'][1]],
                                 [line2['coeffs'][0], line2['coeffs'][1]]])
            B_vector = np.array([line1['rhs'], line2['rhs']])

            det = np.linalg.det(A_matrix)

            if abs(det) < 1e-9: # Parallel or coincident lines, no unique intersection
                continue

            try:
                intersection = np.linalg.solve(A_matrix, B_vector)
                px, py = intersection[0], intersection[1]
                
                # Check if intersection point is within plotting range (with some buffer)
                if not (x.min() - 0.5 <= px <= x.max() + 0.5 and y.min() - 0.5 <= py <= y.max() + 0.5):
                    continue

                # Check if the intersection point satisfies all original constraints (including non-negativity)
                is_feasible_point = True
                if px < -1e-9 or py < -1e-9:
                    is_feasible_point = False
                else:
                    for k, constr_row in enumerate(constraints_data):
                        a_c, b_c, rhs_c = constr_row[0], constr_row[1], constr_row[2]
                        constr_type_c = constraint_types[k]
                        
                        val = a_c * px + b_c * py

                        if constr_type_c == "<=" and val > rhs_c + 1e-6:
                            is_feasible_point = False
                            break
                        elif constr_type_c == ">=" and val < rhs_c - 1e-6:
                            is_feasible_point = False
                            break
                        elif constr_type_c == "=" and not np.isclose(val, rhs_c, atol=1e-6):
                            is_feasible_point = False
                            break

                if is_feasible_point:
                    # Add point if not already added (check for duplicates with tolerance)
                    if not any(np.isclose(p, (px, py), atol=1e-5).all() for p in feasible_points):
                        feasible_points.append((px, py))
                        st.write(f"Feasible intersection point found: **({px:.2f}, {py:.2f})** (from {line1['label']} and {line2['label']}).")

            except np.linalg.LinAlgError:
                continue # Skip if matrix is singular (lines are truly parallel)

    # If no feasible points found, it's either infeasible or unbounded
    if not feasible_points:
        st.error("No feasible region found. The problem might be infeasible or unbounded.")
        st.pyplot(fig)
        return

    # Sort feasible points to draw the polygon for the feasible region 
    try:
        centroid_x = sum([p[0] for p in feasible_points]) / len(feasible_points)
        centroid_y = sum([p[1] for p in feasible_points]) / len(feasible_points)
        feasible_points.sort(key=lambda p: math.atan2(p[1] - centroid_y, p[0] - centroid_x))
    except ZeroDivisionError: # Handle case of single point or no points if this part is reached
        pass

    # Redraw the plot to highlight the feasible region and corner points
    fig_fr, ax_fr = plt.subplots(figsize=(8, 6))
    ax_fr.set_xlabel("x")
    ax_fr.set_ylabel("y")
    ax_fr.set_title("Feasible Region and Corner Points")
    ax_fr.grid(True)
    ax_fr.axvline(0, color='k', linestyle='-', linewidth=0.8)
    ax_fr.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax_fr.set_xlim(0, max_coord)
    ax_fr.set_ylim(0, max_coord)


    # Redraw constraints lines
    for i, constr_row in enumerate(constraints_data):
        a, b, rhs = constr_row[0], constr_row[1], constr_row[2]
        constr_type = constraint_types[i]
        label = f"C{i+1}: {a:.2f}x + {b:.2f}y {constr_type} {rhs:.2f}"
        if b != 0:
            y_line = (rhs - a * x) / b
            ax_fr.plot(x, y_line, linestyle='--', color='gray', alpha=0.7)
        elif a != 0:
            ax_fr.axvline(rhs / a, linestyle='--', color='gray', alpha=0.7)

    # Plot the feasible region with bolder color 
    if len(feasible_points) > 2: 
        poly_x = [p[0] for p in feasible_points]
        poly_y = [p[1] for p in feasible_points]
        ax_fr.fill(poly_x, poly_y, color='purple', alpha=0.4, label='Feasible Region')
    else:
        st.write("Could not form a polygon for the feasible region (less than 3 corner points identified).")

    # Plot feasible points as markers
    for p in feasible_points:
        ax_fr.plot(p[0], p[1], 'o', color='red', markersize=8, markeredgecolor='black')
        ax_fr.annotate(f'({p[0]:.2f}, {p[1]:.2f})', (p[0], p[1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9)

    ax_fr.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    st.pyplot(fig_fr)
    st.markdown("---")

    st.write("### Step 3: Evaluating Objective Function at Corner Points")
    st.write("The **optimal solution** for a Linear Programming problem always occurs at one of the corner points of the feasible region (or along an edge if multiple optimal solutions exist). We evaluate the objective function value at each identified corner point.")
    
    objective_values = []
    for point in feasible_points:
        val = obj_coeffs[0] * point[0] + obj_coeffs[1] * point[1]
        objective_values.append((point, val))

    df_obj_values = {"Point ($x, y$)": [f"({p[0]:.2f}, {p[1]:.2f})" for p, _ in objective_values],
                     "Objective Value (Z)": [f"{val:.2f}" for _, val in objective_values]}
    st.table(df_obj_values)

    optimal_point = None
    optimal_value = -float('inf') if problem_type == "Maximization" else float('inf')

    for point, val in objective_values:
        if problem_type == "Maximization":
            if val > optimal_value:
                optimal_value = val
                optimal_point = point
        else: # Minimization
            if val < optimal_value:
                optimal_value = val
                optimal_point = point

    if optimal_point:
        st.markdown("---")
        st.subheader("Optimal Solution Found!")
        st.success(f"The optimal solution for **{problem_type}** is at **$x={optimal_point[0]:.2f}, y={optimal_point[1]:.2f}$** with an objective value of **$Z={optimal_value:.2f}$**.")
        
        # Highlight optimal point on the graph again
        fig_final, ax_final = plt.subplots(figsize=(8, 6))
        ax_final.set_xlabel("x")
        ax_final.set_ylabel("y")
        ax_final.set_title("Optimal Solution")
        ax_final.grid(True)
        ax_final.axvline(0, color='k', linestyle='-', linewidth=0.8)
        ax_final.axhline(0, color='k', linestyle='-', linewidth=0.8)
        ax_final.set_xlim(0, max_coord)
        ax_final.set_ylim(0, max_coord)

        # Redraw constraints (faded)
        for i, constr_row in enumerate(constraints_data):
            a, b, rhs = constr_row[0], constr_row[1], constr_row[2]
            constr_type = constraint_types[i]
            if b != 0:
                y_line = (rhs - a * x) / b
                ax_final.plot(x, y_line, linestyle='--', color='gray', alpha=0.7)
            elif a != 0:
                ax_final.axvline(rhs / a, linestyle='--', color='gray', alpha=0.7)

        # Redraw feasible region
        if len(feasible_points) > 2:
            poly_x = [p[0] for p in feasible_points]
            poly_y = [p[1] for p in feasible_points]
            ax_final.fill(poly_x, poly_y, color='purple', alpha=0.4, label='Feasible Region')

        # Plot all feasible points
        for p in feasible_points:
            ax_final.plot(p[0], p[1], 'o', color='red', markersize=8, markeredgecolor='black')
        
        # Highlight optimal point
        ax_final.plot(optimal_point[0], optimal_point[1], 'X', color='lime', markersize=15, markeredgecolor='black', label='Optimal Solution')
        ax_final.annotate(f'Optimal: ({optimal_point[0]:.2f}, {optimal_point[1]:.2f})', 
                           (optimal_point[0], optimal_point[1]), textcoords="offset points", 
                           xytext=(5,15), ha='center', fontsize=10, fontweight='bold', color='darkgreen')
        ax_final.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        st.pyplot(fig_final)
    else:
        st.error("Could not determine an optimal solution. Check if the feasible region is empty or unbounded.")


class SimplexSolver:

    def __init__(self, obj_coeffs, constraints_data, constraint_types, problem_type):
        self.obj_coeffs = np.array(obj_coeffs, dtype=float)
        self.constraints_data = np.array(constraints_data, dtype=float)
        self.constraint_types = constraint_types
        self.problem_type = problem_type
        self.M = 10000.0 # A large number for Big M method
        self.num_original_vars = len(obj_coeffs)
        self.num_constraints = len(constraints_data)
        
        self.is_minimization = (problem_type == "Minimization")
        if self.is_minimization:
            self.obj_coeffs *= -1 # Convert minimization to maximization problem

        self.tableau = None
        self.basic_variables = [] 
        self.var_col_map = {}

        self.slack_vars_count = 0
        self.surplus_vars_count = 0
        self.artificial_vars_count = 0
        for c_type in self.constraint_types:
            if c_type == "<=":
                self.slack_vars_count += 1
            elif c_type == ">=":
                self.surplus_vars_count += 1
                self.artificial_vars_count += 1
            elif c_type == "=":
                self.artificial_vars_count += 1

        total_vars_in_tableau = (
            1 + # Z column
            self.num_original_vars +
            self.slack_vars_count +
            self.surplus_vars_count +
            self.artificial_vars_count +
            1 # RHS column
        )
        
        self.tableau = np.zeros((self.num_constraints + 1, total_vars_in_tableau))
        
        self.tableau[0, 0] = 1 # Z coefficient
        # Coefficients of original variables negated for maximization
        self.tableau[0, 1 : 1 + self.num_original_vars] = -self.obj_coeffs

        self.var_col_map["Z"] = 0
        current_col_idx = 1
        for i in range(self.num_original_vars):
            self.var_col_map[f'x{i+1}'] = current_col_idx
            current_col_idx += 1
        
        for i in range(self.slack_vars_count):
            self.var_col_map[f's{i+1}'] = current_col_idx
            current_col_idx += 1
        
        for i in range(self.surplus_vars_count):
            self.var_col_map[f'sur{i+1}'] = current_col_idx # Using 'sur' for surplus
            current_col_idx += 1

        for i in range(self.artificial_vars_count):
            self.var_col_map[f'a{i+1}'] = current_col_idx
            current_col_idx += 1
        
        self.var_col_map["RHS"] = current_col_idx # Last column

        self.basic_variables = ['Z'] + [''] * self.num_constraints 

        current_slack_added = 0
        current_surplus_added = 0
        current_artificial_added = 0

        # Populate constraint rows
        for i in range(self.num_constraints):
            constr_row_data = self.constraints_data[i]
            c_type = self.constraint_types[i]
            
            self.tableau[i+1, 1 : 1 + self.num_original_vars] = constr_row_data[:-1]
            # RHS
            self.tableau[i+1, -1] = constr_row_data[-1]

            # Add slack coef
            if c_type == "<=":
                slack_var_name = f's{current_slack_added+1}'
                self.tableau[i+1, self.var_col_map[slack_var_name]] = 1
                self.basic_variables[i+1] = slack_var_name
                current_slack_added += 1
            elif c_type == ">=":
                surplus_var_name = f'sur{current_surplus_added+1}'
                artificial_var_name = f'a{current_artificial_added+1}'
                self.tableau[i+1, self.var_col_map[surplus_var_name]] = -1
                self.tableau[i+1, self.var_col_map[artificial_var_name]] = 1
                
                # Penalty for artificial variable in objective row
                self.tableau[0, self.var_col_map[artificial_var_name]] = -self.M
                self.basic_variables[i+1] = artificial_var_name
                current_surplus_added += 1
                current_artificial_added += 1
            elif c_type == "=":
                artificial_var_name = f'a{current_artificial_added+1}'
                self.tableau[i+1, self.var_col_map[artificial_var_name]] = 1
                
                # Penalty for artificial variable in objective row
                self.tableau[0, self.var_col_map[artificial_var_name]] = -self.M
                self.basic_variables[i+1] = artificial_var_name
                current_artificial_added += 1
        
        # Construct all column headers for display
        self.col_headers = ["Z"] + [f'x{i+1}' for i in range(self.num_original_vars)] + \
                           [f's{i+1}' for i in range(self.slack_vars_count)] + \
                           [f'sur{i+1}' for i in range(self.surplus_vars_count)] + \
                           [f'a{i+1}' for i in range(self.artificial_vars_count)] + ["RHS"]

        st.write("### Step 1: Convert to Standard Form and Initialize Tableau (Big M Method)")
        st.write("We convert inequalities to equalities by adding **slack variables** (for $\le$), subtracting **surplus variables** and adding **artificial variables** (for $\ge$), and adding **artificial variables** (for $=$).")
        if self.is_minimization:
            st.write(f"Since this is a minimization problem, we convert it to a maximization problem by negating the objective function: $Z' = -Z$.")
            st.write(f"The objective function coefficients become: {self.obj_coeffs.tolist()} (for Max $Z'$).")
        st.write("For artificial variables, a large penalty 'M' is introduced in the objective function ($-\text{M} \times \text{artificial variable}$ for maximization).")
        st.write("The initial tableau is constructed with coefficients of all variables (original, slack, surplus, artificial) and RHS values. The Z-row coefficients for basic artificial variables are made zero by appropriate row operations.")

        st.write("**Initial Tableau (Before adjusting M terms in Z-row for initial basic solutions):**")
        df_initial_tableau = pd.DataFrame(np.round(self.tableau, 4), columns=self.col_headers, index=['Z'] + self.basic_variables[1:])
        st.dataframe(df_initial_tableau)

        # Eliminate M from objective row for artificial variables that are initially basic
        for i in range(self.num_constraints):
            if self.basic_variables[i+1].startswith('a'): 
                art_var_col_idx = self.var_col_map[self.basic_variables[i+1]]
                if self.tableau[0, art_var_col_idx] < -1e-9:
                    st.write(f"Adjusting Z-row for artificial variable `{self.basic_variables[i+1]}` using Row `{i+1}`: $R_0 = R_0 + M \\times R_{i+1}$")
                    self.tableau[0, :] = self.tableau[0, :] + self.M * self.tableau[i+1, :]

        st.write("**Adjusted Initial Tableau:**")
        df_adjusted_tableau = pd.DataFrame(np.round(self.tableau, 4), columns=self.col_headers, index=['Z'] + self.basic_variables[1:])
        st.dataframe(df_adjusted_tableau)
        st.markdown("---")

    def _get_pivot_column(self):
        # Exclude Z and RHS columns
        reduced_obj_row = self.tableau[0, 1 : -1] 
        
        if np.all(reduced_obj_row >= -1e-9): 
            return -1 
        
        # Find the index of the most negative coefficient
        pivot_col_index = np.argmin(reduced_obj_row) + 1 

        return pivot_col_index

    def _get_pivot_row(self, pivot_col_index):

        ratios = []
        for i in range(1, self.num_constraints + 1): # Iterate through constraint rows
            rhs = self.tableau[i, -1]
            pivot_element_val = self.tableau[i, pivot_col_index]

            if pivot_element_val > 1e-9: # Only consider positive pivot elements for valid ratios
                ratios.append(rhs / pivot_element_val)
            else:
                ratios.append(np.inf) # Invalid ratio 

        if all(r == np.inf for r in ratios):
            return -1 # Indicates an unbounded solution

        pivot_row_index = np.argmin(ratios) + 1 
        return pivot_row_index

    def _perform_pivot(self, pivot_row_index, pivot_col_index):
        pivot_element = self.tableau[pivot_row_index, pivot_col_index]
        
        # divide the pivot row by the pivot element to make it 1
        self.tableau[pivot_row_index, :] = self.tableau[pivot_row_index, :] / pivot_element

        # make other elements in pivot column 0
        for i in range(self.num_constraints + 1): 
            if i != pivot_row_index: 
                factor = self.tableau[i, pivot_col_index]
                self.tableau[i, :] = self.tableau[i, :] - factor * self.tableau[pivot_row_index, :]
    
    def solve(self):
        iteration = 0
        
        while True:
            iteration += 1
            st.write(f"### Iteration {iteration}")

            # Select Pivot Column (Entering Variable)
            pivot_col_index = self._get_pivot_column()

            if pivot_col_index == -1:
                st.write("No negative coefficients in the objective row (Row 0). Optimal solution found!")
                break # Optimal solution reached
            
            entering_var_name = self.col_headers[pivot_col_index] 
            st.write(f"**Step {iteration}.1: Select Pivot Column (Entering Variable)**")
            st.write(f"The most negative coefficient in the Z-row is `{self.tableau[0, pivot_col_index]:.4f}` (under `{entering_var_name}`). So, **`{entering_var_name}`** is the entering variable.")
            
            # Select Pivot Row (Leaving Variable)
            pivot_row_index = self._get_pivot_row(pivot_col_index)

            if pivot_row_index == -1:
                st.error("Problem is unbounded. No finite optimal solution exists.")
                return

            st.write(f"**Step {iteration}.2: Select Pivot Row (Leaving Variable)**")
            st.write("We calculate the ratio of the RHS to the corresponding element in the pivot column for each constraint row. We select the row with the smallest positive ratio.")
            
            ratios_display = []
            for i in range(1, self.num_constraints + 1):
                rhs_val = self.tableau[i, -1]
                pivot_element_val = self.tableau[i, pivot_col_index]
                if pivot_element_val > 1e-9: # Only consider positive pivot elements
                    ratio = rhs_val / pivot_element_val
                    ratios_display.append(f"Row {i} (Basic: {self.basic_variables[i]}): {rhs_val:.4f} / {pivot_element_val:.4f} = **{ratio:.4f}**")
                else:
                    ratios_display.append(f"Row {i} (Basic: {self.basic_variables[i]}): Invalid Ratio (non-positive pivot element or division by zero)")
            st.write("\n".join(ratios_display))

            leaving_var_name = self.basic_variables[pivot_row_index]
            st.write(f"The smallest positive ratio determines the pivot row. So, **`{leaving_var_name}`** (Row `{pivot_row_index}`) is the leaving variable.")
            
            # Identify Pivot Element
            st.write(f"**Step {iteration}.3: Identify Pivot Element**")
            pivot_element_value = self.tableau[pivot_row_index, pivot_col_index]
            st.write(f"The pivot element is at the intersection of the pivot row (Row `{pivot_row_index}`) and pivot column (`{entering_var_name}` column), which is **`{pivot_element_value:.4f}`**.")

            # Perform Row Operations
            st.write(f"**Step {iteration}.4: Perform Row Operations**")
            st.write(f"Divide the pivot row (Row `{pivot_row_index}`) by the pivot element (`{pivot_element_value:.4f}`) to make the pivot element 1. Then, use this new pivot row to make all other elements in the pivot column zero through row operations.")
            
            self._perform_pivot(pivot_row_index, pivot_col_index)

            # Update basic variables for the tableau
            self.basic_variables[pivot_row_index] = entering_var_name

            st.write("**Current Tableau:**")
            df_current_tableau = pd.DataFrame(np.round(self.tableau, 4), columns=self.col_headers, index=['Z'] + self.basic_variables[1:])
            st.dataframe(df_current_tableau)
            st.markdown("---")

        st.subheader("Final Solution")
        st.write("The algorithm terminates when all coefficients in the objective row are non-negative.")

        optimal_values = {}
        # Initialize all original variables to 0 (non-basic)
        for var_idx in range(self.num_original_vars):
            optimal_values[f'x{var_idx+1}'] = 0.0

        # Extract values for basic variables
        for i in range(1, self.num_constraints + 1): 
            basic_var = self.basic_variables[i]
            # Check for artificial variable still basic and non-zero
            if basic_var.startswith('a') and self.tableau[i, -1] > 1e-6: # If artificial var is basic and its value is non-zero
                st.error("An artificial variable is basic and non-zero in the final solution. This indicates that the problem has **no feasible solution**.")
                return

            if basic_var.startswith('x'):
                optimal_values[basic_var] = self.tableau[i, -1] # Value is the RHS of its row

        final_obj_value = self.tableau[0, -1]
        if self.is_minimization:
            final_obj_value *= -1 # Convert Z' back to original Z for minimization

        st.success(f"Optimal objective value (Z): **`{final_obj_value:.4f}`**")
        st.write("Optimal values for variables:")
        for var, val in optimal_values.items():
            st.write(f"**`{var}`**: `{val:.4f}`")

        st.markdown("---")


def solve_simplex(obj_coeffs, constraints_data, constraint_types, problem_type):
    st.subheader("Simplex Method Steps")
    solver = SimplexSolver(obj_coeffs, constraints_data, constraint_types, problem_type)
    solver.solve()


def main():
    st.set_page_config(layout="wide")
    st.title("Linear Programming Solver")

    st.sidebar.header("Problem Setup")
    method = st.sidebar.radio("Select Method", ("Graphical (2 variables)", "Simplex (2+ variables)"))
    problem_type = st.sidebar.radio("Problem Type", ("Maximization", "Minimization"))

    default_num_vars = 2 if method == "Graphical (2 variables)" else 4
    num_vars = st.sidebar.number_input("Number of Variables", min_value=2, value=default_num_vars, step=1)
    if method == "Graphical (2 variables)":
        if num_vars != 2:
            st.sidebar.warning("Graphical method only supports 2 variables. Setting to 2.")
            num_vars = 2

    default_num_constraints = 3 if method == "Graphical (2 variables)" or method == "Simplex (2+ variables)" else 1 # Both test cases have 3 constraints
    num_constraints = st.sidebar.number_input("Number of Constraints (excluding non-negativity)", min_value=1, value=default_num_constraints, step=1)

    st.header("Enter Problem Data")

    st.subheader("Objective Function Coefficients")
    obj_coeffs = []
    cols = st.columns(num_vars)
    if method == "Graphical (2 variables)" and num_vars == 2:
        default_obj_coeffs = [2.0, 3.0]
    elif method == "Simplex (2+ variables)" and num_vars == 4:
        default_obj_coeffs = [2.0, 4.0, 1.0, 1.0]
    else:
        default_obj_coeffs = [0.0] * num_vars

    for i in range(num_vars):
        obj_coeffs.append(cols[i].number_input(f"Coeff x{i+1}", value=default_obj_coeffs[i] if i < len(default_obj_coeffs) else 0.0, key=f"obj_coeff_{i}"))

    st.subheader("Constraint Coefficients and RHS")
    constraints_data = []
    constraint_types = []

    # Test Case 1: Graphical Example
    graphical_test_obj = [2.0, 3.0]
    graphical_test_constraints = [[1.0, 1.0, 6.0], [2.0, 1.0, 7.0], [1.0, 4.0, 8.0]]
    graphical_test_types = [">=", ">=", ">="]
    graphical_test_problem = "Minimization"

    # Test Case 2: Simplex Example
    simplex_test_obj = [2.0, 4.0, 1.0, 1.0]
    simplex_test_constraints = [[1.0, 3.0, 0.0, 1.0, 4.0], [2.0, 1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 4.0, 1.0, 3.0]]
    simplex_test_types = ["<=", "<=", "<="]
    simplex_test_problem = "Maximization"
    
    default_constraints = []
    default_constraint_types = []
    if method == "Graphical (2 variables)" and num_vars == 2 and num_constraints == len(graphical_test_constraints):
        default_constraints = graphical_test_constraints
        default_constraint_types = graphical_test_types
    elif method == "Simplex (2+ variables)" and num_vars == 4 and num_constraints == len(simplex_test_constraints):
        default_constraints = simplex_test_constraints
        default_constraint_types = simplex_test_types
    
    for i in range(num_constraints):
        st.write(f"Constraint {i+1}:")
        row_coeffs = []
        cols = st.columns(num_vars + 2) # Coeffs + Type + RHS
        
        for j in range(num_vars):
            default_val = default_constraints[i][j] if i < len(default_constraints) and j < len(default_constraints[i])-1 else 0.0
            row_coeffs.append(cols[j].number_input(f"Coeff x{j+1}", value=default_val, key=f"constr_coeff_{i}_{j}"))
        
        default_inequality_type = default_constraint_types[i] if i < len(default_constraint_types) else "<="
        inequality_type = cols[num_vars].selectbox(f"Type", ("<=", ">=", "="), index=["<=", ">=", "="].index(default_inequality_type), key=f"constr_type_{i}") 
        
        default_rhs_val = default_constraints[i][-1] if i < len(default_constraints) else 0.0
        rhs_val = cols[num_vars+1].number_input(f"RHS", value=default_rhs_val, key=f"constr_rhs_{i}")
        
        # Ensure RHS is non-negative by multiplying by -1 if negative, and flipping inequality sign
        if rhs_val < 0:
            st.warning(f"Constraint {i+1}: RHS is negative. Multiplying entire constraint by -1 and flipping inequality sign.")
            rhs_val *= -1
            row_coeffs = [-c for c in row_coeffs]
            if inequality_type == "<=":
                inequality_type = ">="
            elif inequality_type == ">=":
                inequality_type = "<="
            st.write(f"Modified constraint {i+1}: {' + '.join([f'{c}x{idx+1}' for idx, c in enumerate(row_coeffs)])} {inequality_type} {rhs_val}")

        constraints_data.append(row_coeffs + [rhs_val])
        constraint_types.append(inequality_type)

    if st.button("Solve LP"):
        st.subheader("Problem Summary")
        obj_str = f"{'Max' if problem_type == 'Maximization' else 'Min'} Z = " + " + ".join([f"{c}x{i+1}" for i, c in enumerate(obj_coeffs)])
        st.write(obj_str)
        st.write("Subject to:")
        for i, constr in enumerate(constraints_data):
            constr_str = " + ".join([f"{c}x{j+1}" for j, c in enumerate(constr[:-1])])
            st.write(f"{constr_str} {constraint_types[i]} {constr[-1]}")
        st.write("$x_i \ge 0$ for all $i$")

        st.markdown("---")
        st.subheader("Solution Steps")

        is_graphical_test_case_match = (
            method == "Graphical (2 variables)" and
            problem_type == graphical_test_problem and
            num_vars == len(graphical_test_obj) and
            num_constraints == len(graphical_test_constraints) and
            np.allclose(obj_coeffs, graphical_test_obj) and
            len(constraints_data) == len(graphical_test_constraints) and # Ensure lengths match
            all(
                np.allclose(np.array(c[:-1]), np.array(tc[:-1])) and
                np.isclose(c[-1], tc[-1]) and
                constraint_types[idx] == graphical_test_types[idx]
                for idx, (c, tc) in enumerate(zip(constraints_data, graphical_test_constraints))
            )
        )

        is_simplex_test_case_match = (
            method == "Simplex (2+ variables)" and
            problem_type == simplex_test_problem and
            num_vars == len(simplex_test_obj) and
            num_constraints == len(simplex_test_constraints) and
            np.allclose(obj_coeffs, simplex_test_obj) and
            len(constraints_data) == len(simplex_test_constraints) and # Ensure lengths match
            all(
                np.allclose(np.array(c[:-1]), np.array(tc[:-1])) and
                np.isclose(c[-1], tc[-1]) and
                constraint_types[idx] == simplex_test_types[idx]
                for idx, (c, tc) in enumerate(zip(constraints_data, simplex_test_constraints))
            )
        )

        if is_graphical_test_case_match:
            st.info("Recognized graphical test case. Solving automatically...")
            solve_graphical(graphical_test_obj, graphical_test_constraints, graphical_test_types, graphical_test_problem)
        elif is_simplex_test_case_match:
            st.info("Recognized simplex test case. Solving automatically...")
            solve_simplex(simplex_test_obj, simplex_test_constraints, simplex_test_types, simplex_test_problem)
        elif method == "Graphical (2 variables)":
            if num_vars == 2:
                solve_graphical(obj_coeffs, constraints_data, constraint_types, problem_type)
            else:
                st.error("Graphical method is only for 2 variables. Please adjust the 'Number of Variables' in the sidebar.")
        else: # Simplex Method
            solve_simplex(obj_coeffs, constraints_data, constraint_types, problem_type)

if __name__ == "__main__":
    main()