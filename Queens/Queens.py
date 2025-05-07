import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
import sys
from collections import defaultdict
from time import time

def parse_board(filename):
    with open(filename, 'r') as f:
        return [list(map(int, line.strip().split(','))) for line in f if line.strip()]


def dfs(board):
    assignment_history = [[-1 for i in range(len(board))]]
    steps = 0
    start_time = time()
    def dfs_helper(row, n, board, queen_positions, used_columns, used_colors):
        nonlocal steps
        if row == n:
            assignment_history.append(queen_positions.copy())
            return queen_positions

        for col in range(n):

            isValid = True
            if col in used_columns or board[row][col] in used_colors: isValid = False
            for r2, c2 in enumerate(queen_positions[:row]):
                if abs(r2 - row) <= 1 and abs(c2 - col) <= 1:
                    isValid = False
                    break

            if isValid:
                queen_positions[row] = col
                used_columns.add(col)
                used_colors.add(board[row][col])
                assignment_history.append(queen_positions.copy())
                steps += 1
                result = dfs_helper(row + 1, n, board, queen_positions, used_columns, used_colors)
                if result:
                    return queen_positions

                # Backtrack
                queen_positions[row] = -1
                used_columns.remove(col)
                used_colors.remove(board[row][col])
                assignment_history.append(queen_positions.copy())

        return False

    return dfs_helper(0, len(board), board, [-1 for i in range(len(board))], set(), set()), steps, time()-start_time, assignment_history


def dfsplus(board):
    assignment_history = [[-1 for i in range(len(board))]]
    steps = 0
    start_time = time()
    def dfs_helper(queens_left, board, queen_positions, queen_domains, used_colors):
        nonlocal steps 

        if not queens_left:
            assignment_history.append(queen_positions.copy())
            return queen_positions

        queen = min(queens_left, key= lambda x: len(queen_domains[x]))
        queens_left.remove(queen)


        for col in queen_domains[queen]:
            if board[queen][col] not in used_colors: #validity check that current forward checking doesn't handle
                queen_positions[queen] = col

                #forward check
                removed_domains = defaultdict(set)

                # x-ing out column for other rows
                for q in queens_left:
                    if col in list(queen_domains[q]):
                        queen_domains[q].remove(col)
                        removed_domains[q].add(col)
                
                # x-ing out corners of [queen][col] placement in other parts of the board
                for q in queens_left:
                    if abs(q - queen) <= 1:  # Only rows that are adjacent
                        for c in list(queen_domains[q]):
                            if abs(c - col) <= 1:
                                queen_domains[q].remove(c)
                                removed_domains[q].add(c)

                isValid = True
                for q in queens_left:
                    if not (queen_domains[q]): 
                        for q in queens_left: queen_domains[q].update(removed_domains[q]) #undo forward check
                        isValid = False
                if not isValid: continue

                used_colors.add(board[queen][col])
                assignment_history.append(queen_positions.copy())
                steps += 1

                
                result = dfs_helper(queens_left, board, queen_positions, queen_domains, used_colors)
                if result: return queen_positions

                # Backtrack
                queen_positions[queen] = -1
                for q in queens_left: queen_domains[q].update(removed_domains[q]) #undo forward check
                used_colors.remove(board[queen][col])
                assignment_history.append(queen_positions.copy())

        queens_left.add(queen)

        return False

    return dfs_helper(set([i for i in range(len(board))]), board, [-1 for i in range(len(board))], {j:set([i for i in range(len(board))]) for j in range(len(board))}, set()), steps, time()-start_time, assignment_history



def dfsplusplus(board):
    assignment_history = [[-1 for i in range(len(board))]]
    start_time = time()
    steps = 0
    def dfs_helper(queens_left, queen_positions, queen_domains):
        nonlocal steps
        if not queens_left:
            assignment_history.append(queen_positions.copy())
            return queen_positions

        queen = min(queens_left, key= lambda x: len(queen_domains[x]))
        queens_left.remove(queen)
        current_color = queen

        for row, col in queen_domains[queen]:
                queen_positions[row] = col

                removed_domains = defaultdict(set)
                #print(queen_domains[queen])
                for c2 in range(n):
                    if c2 == col: continue
                    color = board[row][c2]
                    if color == current_color or color not in queens_left: continue
                    if (row, c2) in queen_domains[color]:
                        queen_domains[color].remove((row, c2))
                        removed_domains[color].add((row, c2))

                # Prune same column
                for r2 in range(n):
                    if r2 == row: continue
                    color = board[r2][col]
                    if color == current_color or color not in queens_left: continue
                    if (r2, col) in queen_domains[color]:
                        queen_domains[color].remove((r2, col))
                        removed_domains[color].add((r2, col))

                # Prune adjacent corners
                for dr in [-1, 1]:
                    for dc in [-1, 1]:
                        r2, c2 = row + dr, col + dc
                        if 0 <= r2 < n and 0 <= c2 < n:
                            color = board[r2][c2]
                            if color == current_color or color not in queens_left: continue
                            if (r2, c2) in queen_domains[color]:
                                queen_domains[color].remove((r2, c2))
                                removed_domains[color].add((r2, c2))
                
                isValid = True
                for q in queens_left:
                    if not (queen_domains[q]): 
                        for q in queens_left: queen_domains[q].update(removed_domains[q]) #undo forward check
                        isValid = False
                if not isValid: continue


                assignment_history.append(queen_positions.copy())
                steps += 1
                result = dfs_helper(queens_left, queen_positions, queen_domains)
                if result: return queen_positions

                # Backtrack
                queen_positions[row] = -1
                for q in queens_left: queen_domains[q].update(removed_domains[q]) #undo forward check
                assignment_history.append(queen_positions.copy())

        queens_left.add(queen)

        return False

    n = len(board)
    #Each queens domain is it's color
    queen_domains = [set() for i in range(n)]
    for row in range(n):
        for col in range(n):
            queen_domains[board[row][col]].add((row,col))

    return dfs_helper(set([i for i in range(n)]), [-1 for i in range(n)], queen_domains), steps, time()-start_time, assignment_history



def animate_queens(frames, board):
    print(frames)
    print("Animating ", len(frames), " steps!")
    n = len(board)
    fig, ax = plt.subplots(figsize=(n, n))
    board_array = np.array(board)

    # Generate a unique color for each number
    unique_vals = sorted(set(val for row in board for val in row))
    cmap = plt.get_cmap('tab20', len(unique_vals))
    color_map = {val: cmap(i) for i, val in enumerate(unique_vals)}

    def update(frame_index):
        ax.clear()
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Step {frame_index+1}")

        # Draw colored board cells
        for r in range(n):
            for c in range(n):
                color = color_map[board[r][c]]
                ax.add_patch(plt.Rectangle((c, n - r - 1), 1, 1, color=color))

        frame = frames[frame_index]
        queen_positions = [(r, col) for r, col in enumerate(frame) if col != -1]

        # Draw Xs in rows, columns, and corners for placed queens
        for qr, qc in queen_positions:
            for r in range(n):
                if (r, qc) not in queen_positions:
                    ax.text(qc + 0.5, n - r - 0.5, "X", ha='center', va='center', fontsize=14, color='darkgrey')
            for c in range(n):
                if (qr, c) not in queen_positions:
                    ax.text(c + 0.5, n - qr - 0.5, "X", ha='center', va='center', fontsize=14, color='darkgrey')
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    rr, cc = qr + dr, qc + dc
                    if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in queen_positions:
                        ax.text(cc + 0.5, n - rr - 0.5, "X", ha='center', va='center', fontsize=14, color='darkgrey')

        # Draw queens
        for r, col in queen_positions:
            ax.add_patch(plt.Circle((col + 0.5, n - r - 0.5), 0.3, color='black'))

        # Grid lines
        for i in range(n + 1):
            ax.plot([0, n], [i, i], color='black')
            ax.plot([i, i], [0, n], color='black')


    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)
    plt.show()




def main():
    
    if len(sys.argv) != 3:
        print("Usage: python queens_solver.py <input_file> <0 for DFSB | 1 for DFSBplus | 2 for DFSplusplus>")
        return

    input_path = sys.argv[1]
    #output_path = sys.argv[2]
    method = int(sys.argv[2])

    board = parse_board(input_path)

    if method == 0:
        solution, _, _, assignment_history = dfs(board)
    elif method == 1:
        solution, _, _, assignment_history = dfsplus(board)
    else:
        solution, _, _, assignment_history = dfsplusplus(board)


    '''
    if not solution:
        with open(output_path, 'w') as f:
            f.write("No answer.\n")
        print("❌ No solution found.")
    else:
        with open(output_path, 'w') as f:
            for row, col in enumerate(solution):
                f.write(f"{row},{col}\n")
        print("✅ Solution written to file.")
    '''
    
    animate_queens(assignment_history, board)
    

    '''
    data = []
    for i in range(1, 58):
        print(i)
        filename = "games/queens"+str(i)+".txt"
        time, time_h, time_hh = None, None, None
        steps, steps_h, steps_hh = None, None, None
        for h in range(0, 3):
            board = parse_board(filename)
            if h == 0:
                solution, s, t = dfs(board) #assignment_history
                time = t
                steps = s
            elif h == 1:
                solution, s, t = dfsplus(board) #assignment_history
                time_h = t
                steps_h = s
            else:
                solution, s, t = dfsplusplus(board) #assignment_history
                time_hh = t
                steps_hh = s

            if solution == None: print("Couldn't solve ", i, h)
        data.append({"h":h, "size":len(board), "time":time, "time_h":time_h, "time_hh":time_hh, "steps":steps, "steps_h":steps_h, "steps_hh":steps_hh})

    print(data)
    '''
main()