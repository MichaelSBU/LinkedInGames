# Here's the full implementation of the Tango CSP solver as a script-style module.

import sys
from collections import defaultdict
from random import randrange
import sys
import copy
from time import time

sys.setrecursionlimit(2500)  # Handle deep recursion for backtracking


def parse_tango_input(file_path):
    grid = []
    constraints = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    n = len(lines[0].strip().split(','))

    # Grid parsing (first n lines)
    for i in range(n):
        row = lines[i].strip().split(',')
        grid.append([-1 if cell == '' else int(cell)-1 for cell in row])

    # Pair constraints (= or x)
    pair_constraints = []
    for line in lines[n:]:
        if line.strip() == "":
            continue
        parts = line.strip().replace('=', ',=,').replace('x', ',x,').replace('×', ',x,').split(',')
        r1, c1, rel, r2, c2 = int(parts[0]), int(parts[1]), parts[2], int(parts[3]), int(parts[4])
        pair_constraints.append(((r1, c1), (r2, c2), rel))

    #for z in grid:
    #    print(z)
    #print(pair_constraints)
    return grid, pair_constraints, n


def build_csp(grid, pair_constraints, n):
    variables = []
    assignments = {}
    domain = [0, 1]
    constraints = defaultdict(set)
    adjacency = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for r in range(n):
        for c in range(n):
            if grid[r][c] == -1:
                variables.append((r, c))
                assignments[(r, c)] = -1
            else:
                assignments[(r, c)] = grid[r][c]

    # Enforce row and column count constraints (3 suns and 3 moons)
    row_counts = {r: [0, 0] for r in range(n)}  # [moons, suns]
    col_counts = {c: [0, 0] for c in range(n)}

    for r in range(n):
        for c in range(n):
            val = assignments[(r, c)]
            if val != -1:
                row_counts[r][val] += 1
                col_counts[c][val] += 1

    for r in range(n):
        for c in range(n):
            var = (r, c)
            if assignments[var] != 0:
                continue
            # neighbors for no-three-in-a-row constraint
            for dr, dc in adjacency:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    constraints[var].add((nr, nc))

    # Add pair constraints
    pair_constraint_map = {}
    for (r1, c1), (r2, c2), rel in pair_constraints:
        constraints[(r1, c1)].add((r2, c2))
        constraints[(r2, c2)].add((r1, c1))
        pair_constraint_map[((r1, c1), (r2, c2))] = rel
        pair_constraint_map[((r2, c2), (r1, c1))] = rel

    return variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map


def is_valid(assignments, row_counts, col_counts, pair_constraint_map, n):
    for r in range(n):
        for c in range(n):
            val = assignments[(r, c)]
            if val == -1: continue
            # Debugging: Print current assignments
            #print(f"Checking cell ({r}, {c}): {val}")
            
            # No three in a row/column
            if r >= 2 and assignments[(r - 1, c)] == val and assignments[(r - 2, c)] == val: 
                #print(f"Invalid: three in a row at ({r}, {c})")
                return False
            if c >= 2 and assignments[(r, c - 1)] == val and assignments[(r, c - 2)] == val: 
                #print(f"Invalid: three in a column at ({r}, {c})")
                return False
            
            # Pair constraints
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    rel = pair_constraint_map.get(((r, c), (nr, nc)))
                    if rel == '=' and assignments[(r, c)] != -1 and assignments[(nr, nc)] != -1 and assignments[(r, c)] != assignments[(nr, nc)]:
                        #print(f"Invalid: pair constraint '=' violated at ({r}, {c}) and ({nr}, {nc})")
                        return False
                    if rel == 'x' and assignments[(r, c)] != -1 and assignments[(nr, nc)] != -1 and assignments[(r, c)] == assignments[(nr, nc)]:
                        #print(f"Invalid: pair constraint 'x' violated at ({r}, {c}) and ({nr}, {nc})")
                        return False
    for i in range(n):
        if row_counts[i][0] > n / 2 or row_counts[i][1] > n / 2: 
            #print(f"Invalid: row count constraint violated at row {i}")
            return False
        if col_counts[i][0] > n / 2 or col_counts[i][1] > n / 2: 
            #print(f"Invalid: column count constraint violated at column {i}")
            return False

    #print("Is valid")
    return True


def DFSB(variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map, n):
    assignment_history = [assignments.copy()]  # List to track the assignment at each level of recursion
    start_time = time()
    steps = 0
    
    def helper(vars_left):
        nonlocal steps
        # Base case: if no variables left, check if the current assignment is valid
        if not vars_left:
            assignment_history.append(assignments.copy())  # Track the valid solution
            return assignments.copy()

        # Get the next variable to assign
        var = vars_left.pop()
        r, c = var
        
        for val in domain:
            #print(val)
            if row_counts[r][val] >= n/2 or col_counts[c][val] >= n/2:
                continue  # Skip if assigning this value would violate row/column count

            # Assign the value to the variable
            assignments[var] = val
            row_counts[r][val] += 1
            col_counts[c][val] += 1

            # Check if the assignment is valid
            if is_valid(assignments, row_counts, col_counts, pair_constraint_map, n):
                assignment_history.append(assignments.copy())  # Track the assignment at this level
                steps += 1
                result = helper(vars_left)  # Recursively attempt to assign the remaining variables
                if result:
                    return result  # Return the solution if found

            # Backtrack: undo the assignment if it leads to an invalid state
            assignments[var] = -1
            row_counts[r][val] -= 1
            col_counts[c][val] -= 1

        # If no valid assignment found, add the variable back to the list and return False
        vars_left.append(var)
        return False

   
    return helper(variables.copy()), time()-start_time, steps, assignment_history


def print_board(assignments, variable_domains, n):
    """Prints the current board with assigned values and domains of unassigned variables."""
    for r in range(n):
        row = ""
        for c in range(n):
            var = (r, c)
            if assignments[var] != -1:
                row += f"{assignments[var]} ".ljust(4)
            else:
                # For unassigned variables, print the domain in []
                domain_str = "".join([str(d) for d in sorted(variable_domains[var])])
                row += f"[{domain_str}] ".ljust(8)
        print(row)
    print("\n")




def violates_triple_rule(assignments, var, n):
    r, c = var
    val = assignments[var]

    def check_line(line):
        for i in range(len(line) - 2):
            if line[i] == line[i+1] == line[i+2] != -1:
                return True
        return False

    row_vals = [assignments.get((r, j), -1) for j in range(n)]
    col_vals = [assignments.get((i, c), -1) for i in range(n)]

    return check_line(row_vals) or check_line(col_vals)

def DFSBplusplus(variables, assignments, full_domain, constraints, row_counts, col_counts, pair_constraint_map, n):
    assignment_history = [assignments.copy()]
    variable_domains = {var: set(full_domain) for var in variables if assignments[var] == -1}
    start_time = time()
    steps = 0
    # Prune domains based on the current assignments
    for (r, c), val in assignments.items():
        if val != -1:  # Only consider assigned variables
            # Prune row and column for each assigned variable
            for var in variables:
                #if var == (2,5): 
                #    print('\n')
                #    print(variable_domains[var])
                #    print(r,c)
                if var != (r, c) and assignments[var] == -1:  # Skip already assigned variables
                    nr, nc = var
                    # Row constraint pruning: If val is in this row, prune it from the other unassigned variables in the row
                    if (row_counts[r][val] >= n // 2 and r == var[0]) or (col_counts[c][val] >= n // 2 and c == var[1]):
                        if val in variable_domains[var]:
                            #if var == (2,5): print("HEREHHEHEHGEHEHEHHEHEHEH", r, c, var, val)
                            variable_domains[var].remove(val)

                    #Triple check pruning
                    assignments[var] = val  # Assign val directly to assignments
                    if violates_triple_rule(assignments, var, n):
                        if val in variable_domains[var]:
                            variable_domains[var].remove(val)
                    assignments[var] = -1  # Reset the assignment back to -1


                    #Pairwise constraints for neighbors
                    if (r, c) in pair_constraint_map:
                        for (r2, c2), constraint in pair_constraint_map[(r, c)].items():
                            if var == (r2, c2):
                                if constraint == '=' and val != assignments.get(var, -1):
                                    if val in variable_domains[var]:
                                        variable_domains[var].remove(val)
                                elif constraint == 'x' and val == assignments.get(var, -1):
                                    if val in variable_domains[var]:
                                        variable_domains[var].remove(val)
                    #if var == (2,5): 
                    #    print(variable_domains[var])
                    #    print(r,c)
    #print(variable_domains[(2,5)])
    def helper(vars_left, variable_domains):
        nonlocal steps
        #print_board(assignments, variable_domains, n)
        if not vars_left:
                assignment_history.append(assignments.copy())
                return assignments.copy()

        var = min(vars_left, key=lambda var: len(variable_domains[var]))
        vars_left.remove(var)
        r, c = var
        #print("here")
        #print(r,c)
        #print(variable_domains[var])
        for val in variable_domains[var]:
            #print("val",val)
            if row_counts[r][val] >= n // 2 or col_counts[c][val] >= n // 2:
                continue

            assignments[var] = val
            row_counts[r][val] += 1
            col_counts[c][val] += 1
            assignment_history.append(assignments.copy())
            steps += 1

            removed = defaultdict(set)

            # Row pruning
            if row_counts[r][val] == n // 2:
                if row_counts[r][1 - val] == n // 2 - 1:
                    for j in range(n):
                        cell = (r, j)
                        if assignments.get(cell, -1) == -1 and cell in variable_domains:
                            if val in variable_domains[cell]:
                                variable_domains[cell].remove(val)
                                removed[cell].add(val)

            # Column pruning
            if col_counts[c][val] == n // 2:
                if col_counts[c][1 - val] == n // 2 - 1:
                    for i in range(n):
                        cell = (i, c)
                        if assignments.get(cell, -1) == -1 and cell in variable_domains:
                            if val in variable_domains[cell]:
                                variable_domains[cell].remove(val)
                                removed[cell].add(val)


            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                for step in range(1, 3):
                    nr, nc = r + dr * step, c + dc * step
                    if 0 <= nr < n and 0 <= nc < n:
                        neighbor = (nr, nc)
                        if neighbor in vars_left:
                            to_remove = set()

                            for d in variable_domains[neighbor]:
                                # Constraint 1: row/col balance
                                if row_counts[nr][d] >= n // 2 or col_counts[nc][d] >= n // 2:
                                    to_remove.add(d)

                                # Constraint 2: triple rule
                                assignments[neighbor] = d
                                if violates_triple_rule(assignments, neighbor, n): to_remove.add(d)
                                assignments[neighbor] = -1  # undo temp assignment

                                # Constraint 3: pairwise constraints (only check when step == 1)
                                if step == 1:
                                    constraint = pair_constraint_map.get((var, neighbor))
                                    if constraint == '=' and d != val:
                                        to_remove.add(d)
                                    elif constraint == 'x' and d == val:
                                        to_remove.add(d)

                                    constraint = pair_constraint_map.get((neighbor, var))
                                    if constraint == '=' and d != val:
                                        to_remove.add(d)
                                    elif constraint == 'x' and d == val:
                                        to_remove.add(d)

                            for d in to_remove:
                                if d in variable_domains[neighbor]:
                                    variable_domains[neighbor].remove(d)
                                    removed[neighbor].add(d)

            if all(variable_domains[v] for v in vars_left):
                result = helper(vars_left, variable_domains)
                if result:
                    return result

            for v, vals in removed.items(): variable_domains[v].update(vals)
            assignments[var] = -1
            row_counts[r][val] -= 1
            col_counts[c][val] -= 1

        vars_left.append(var)
        return False

    result = helper(list(variable_domains.keys()), variable_domains)
    return result,  time()-start_time, steps, assignment_history








import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate(assignment_history, pair_constraint_map, n, repeat_count=1000):
    print("Animating", len(assignment_history), "steps!")
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    first_frame_assignments = assignment_history[0]
    total_frames = len(assignment_history)
    total_plays = repeat_count * total_frames

    def update_grid(i):
        frame = i % total_frames
        loop = i // total_frames + 1
        ax.clear()
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Draw grid lines
        for r in range(n + 1):
            ax.plot([0, n], [r, r], color='gray', linewidth=1.5)
            ax.plot([r, r], [0, n], color='gray', linewidth=1.5)

        # Draw cells
        for r in range(n):
            for c in range(n):
                value = assignment_history[frame].get((r, c), None)
                if first_frame_assignments[(r, c)] != -1:
                    ax.add_patch(plt.Rectangle((c, n - r - 1), 1, 1, color='lightgray'))
                    if value == 0:
                        ax.add_patch(plt.Circle((c + 0.5, n - r - 0.5), 0.3, color='orange'))
                    elif value == 1:
                        ax.add_patch(plt.Circle((c + 0.5, n - r - 0.5), 0.3, color='blue'))
                        ax.add_patch(plt.Circle((c + 0.35, n - r - 0.4), 0.22, color='lightgray'))
                else:
                    if value == 0:
                        ax.add_patch(plt.Circle((c + 0.5, n - r - 0.5), 0.3, color='orange'))
                    elif value == 1:
                        ax.add_patch(plt.Circle((c + 0.5, n - r - 0.5), 0.3, color='blue'))
                        ax.add_patch(plt.Circle((c + 0.35, n - r - 0.4), 0.22, color='white'))

        # Constraints
        for r in range(n):
            for c in range(n):
                if c < n - 1:
                    constraint = pair_constraint_map.get(((r, c), (r, c + 1)))
                    if constraint in {'=', 'x'}:
                        ax.text(c + 1, n - r - 0.5, constraint, color='dimgray',
                                ha='center', va='center', fontsize=16, fontweight="bold")
                if r < n - 1:
                    constraint = pair_constraint_map.get(((r, c), (r + 1, c)))
                    if constraint in {'=', 'x'}:
                        ax.text(c + 0.5, n - r - 1.0, constraint, color='dimgray',
                                ha='center', va='center', fontsize=16, fontweight="bold")

        ax.set_title(f"Step {frame + 1} of {total_frames} (Loop #: {loop})")

    ani = animation.FuncAnimation(fig, update_grid, frames=total_plays, repeat=False, interval=300)
    plt.show()








def main():
    
    
    if len(sys.argv) != 3:
        print("Usage: python tango_solver.py <input_file>  <0 for DFSB | 1 for DFSBplusplus>")
        return

    input_path = sys.argv[1]
    method = int(sys.argv[2])

    grid, pair_constraints, n = parse_tango_input(input_path)
    variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map = build_csp(grid, pair_constraints, n)

    if method == 0:
        solution, time, steps, assignment_history = DFSB(variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map, n)
    else:
        solution, time, steps, assignment_history = DFSBplusplus(variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map, n)

    '''
    if not solution:
        print("NOT SOLVABLE")
        with open(output_path, 'w') as f:
            f.write("No answer.")
    else:
        with open(output_path, 'w') as f:
            for r in range(n):
                line = ",".join(str(assignments[(r, c)]) for c in range(n))
                f.write(line + "\n")
    '''
    animate(assignment_history, pair_constraint_map, n)
    

    '''
    data = []
    for i in range(1, 59):
        print(i)
        filename = "games/tango"+str(i)+".txt"
        for h in range(1, 3):
            grid, pair_constraints, n = parse_tango_input(filename)
            variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map = build_csp(grid, pair_constraints, n)
            fill = 0
            for z in assignments:
                if assignments[z] != -1:
                        fill += 1
            print(h)
            if h == 1:
                solution, time, steps, assignment_history = DFSB(variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map, n)
            else:
                solution, time, steps, assignment_history = DFSBplusplus(variables, assignments, domain, constraints, row_counts, col_counts, pair_constraint_map, n)

            if solution == None: print("Couldn't solve ", i, h)
            data.append({"h":h, "constraints":len(constraints), "fill":fill, "time":time, "steps":steps})

    print(data)
    '''
main()