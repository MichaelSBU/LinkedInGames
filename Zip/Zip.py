import sys
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.collections import LineCollection
from collections import deque
from time import time

DIRS = {(-1, 0):'U', (1, 0):'D', (0, -1):'L', (0, 1):'R'}
DIRS_LIST = list(DIRS.keys())

class ZipSolver:
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.target_sequence = self._extract_target_sequence()
        self.number_to_pos = self._map_numbers_to_positions()
        self.animation_frames = []
        self.time = 0

    def _map_numbers_to_positions(self):
        pos_map = {}
        for i in range(self.n):
            for j in range(self.n):
                val = self.board[i][j]
                if val in self.target_sequence:
                    pos_map[val] = (i, j)
        return pos_map

    def _extract_target_sequence(self):
        numbers = sorted(set(val for row in self.board for val in row if val > 0))
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            raise ValueError(f"Board numbers must be consecutive from 1 to {len(numbers)}: got {numbers}")
        return expected


    def in_bounds(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    def solve(self, h):
        start = self.number_to_pos[1]
        visited = set([start])
        t = time()
        if not h:
            p =  self._dfs(start, visited, 1, [start])
        else:
            #if h == 1:
            #    p =  self._a_star(start, self.h1)
            if h == 2:
                p = self._a_star(start, self.h2)
            elif h == 3:
                p = self._a_star(start, self.h3)
            elif h == 4:
                p = self._a_star(start, self.h4)
            elif h == 5:
                p = self._a_star(start, self.h5)
            elif h == 6:
                p = self._a_star(start, self.h6)
            elif h == 7:
                p = self._a_star(start, self.h7)
            elif h == 8:
                p = self._a_star(start, self.h8)
            elif h == 9:
                p = self._a_star(start, self.h9)
            elif h == 10:
                p = self._a_star(start, self.h10)
            
        self.time = time()-t
        return p

    def _dfs(self, pos, visited, current_number, path):
        self.animation_frames.append((list(path), True))

        if self.board[pos[0]][pos[1]] == self.target_sequence[-1] and len(visited) == self.n * self.n:
            return list(path)

        for dx, dy in DIRS_LIST:
            nx, ny = pos[0] + dx, pos[1] + dy
            next_pos = (nx, ny)

            if not self.in_bounds(nx, ny) or next_pos in visited:
                continue

            tile_value = self.board[nx][ny]
            if tile_value != 0 and tile_value != current_number+1: continue
            next_number = current_number + 1 if tile_value == current_number + 1 else current_number

            self.animation_frames.append((list(path), False))

            visited.add(next_pos)
            path.append(next_pos)

            result = self._dfs(next_pos, visited, next_number, path)
            if result:
                return result

            # backtrack
            visited.remove(next_pos)
            path.pop()

        return None

    def h1(self, pos, visited, current_number):
        # 1. Number order progress
        number_progress = self.target_sequence[-1] - current_number

        # 2. Remaining unvisited clusters (to encourage filling everything)
        def count_clusters():
            visited_set = set(visited)
            clusters = 0
            seen = set()

            for x in range(self.n):
                for y in range(self.n):
                    if (x, y) in visited_set or (x, y) in seen:
                        continue
                    clusters += 1
                    # flood fill
                    queue = [(x, y)]
                    while queue:
                        cx, cy = queue.pop()
                        if (cx, cy) in seen or (cx, cy) in visited_set:
                            continue
                        seen.add((cx, cy))
                        for dx, dy in DIRS_LIST:
                            nx, ny = cx + dx, cy + dy
                            if self.in_bounds(nx, ny) and (nx, ny) not in seen and (nx, ny) not in visited_set:
                                queue.append((nx, ny))
            return clusters

        cluster_penalty = count_clusters()
        #print(number_progress, cluster_penalty)
        return number_progress + cluster_penalty

    def h2(self, pos, visited, current_number):

        # 1. Number order progress
        number_progress = self.target_sequence[-1] - current_number

        # 2. MST over unvisited cells (to encourage board coverage efficiently)
        unvisited = [
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ]

        if not unvisited:
            return number_progress  # all cells visited

        # Prim's algorithm for MST over unvisited tiles
        def compute_mst(points):
            mst_cost = 0
            visited_nodes = set()
            heap = []

            # Start from an arbitrary first point
            visited_nodes.add(points[0])
            for other in points[1:]:
                cost = abs(points[0][0] - other[0]) + abs(points[0][1] - other[1])
                heapq.heappush(heap, (cost, points[0], other))

            while len(visited_nodes) < len(points):
                while True:
                    cost, frm, to = heapq.heappop(heap)
                    if to not in visited_nodes: break
                mst_cost += cost
                visited_nodes.add(to)
                for other in points:
                    if other not in visited_nodes:
                        new_cost = abs(to[0] - other[0]) + abs(to[1] - other[1])
                        heapq.heappush(heap, (new_cost, to, other))

            return mst_cost

        mst_cost = compute_mst(unvisited)
        #print(number_progress, mst_cost)
        #print(mst_cost, len(unvisited))
        return number_progress + mst_cost

    def is_connected(self, unvisited):
        # Use BFS to check if all unvisited cells are connected
        if not unvisited:
            return True

        # Start from the first unvisited cell
        for start in unvisited: break
        visited_set = set()
        queue = [start]
        visited_set.add(start)

        # Directions for 4-connected neighbors (left, right, up, down)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.pop(0)
            # Check all 4 neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in unvisited and (nx, ny) not in visited_set:
                    visited_set.add((nx, ny))
                    queue.append((nx, ny))

        # If the size of visited_set is equal to the size of unvisited, they are connected
        return len(visited_set) == len(unvisited)


    def h3(self, pos, visited, current_number):
        # 1. Number order progress
        number_progress = self.target_sequence[-1] - current_number

        # 2. Identify unvisited cells
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])

        if not unvisited:
            return number_progress  # All cells are visited, no unvisited cells left

        # If unvisited cells are not connected, return infinity
        if not self.is_connected(unvisited):
            return float('inf')


        # 5. Return the heuristic: number progress + MST cost
        return number_progress + len(unvisited)


    def h4(self, pos, visited, current_number):
        number_progress = self.target_sequence[-1] - current_number
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])
        if not self.is_connected(unvisited):return float('inf')
        return number_progress 


    def h5(self, pos, visited, current_number):
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])
        if not self.is_connected(unvisited):return float('inf')
        return len(unvisited)


    def h6(self, pos, visited, current_number):
        number_progress = self.target_sequence[-1] - current_number
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])
        if not unvisited:
            return 2*number_progress 

        if not self.is_connected(unvisited):
            return float('inf')
        return 2*number_progress + len(unvisited)

    def h7(self, pos, visited, current_number):
        number_progress = self.target_sequence[-1] - current_number
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])
        if not unvisited:
            return 3*number_progress 

        if not self.is_connected(unvisited):
            return float('inf')
        return 3*number_progress + len(unvisited)

    def h8(self, pos, visited, current_number):
        number_progress = self.target_sequence[-1] - current_number
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])
        if not unvisited:
            return number_progress 
        if not self.is_connected(unvisited):
            return float('inf')
        return number_progress + 2*len(unvisited)

    def h9(self, pos, visited, current_number):
        number_progress = self.target_sequence[-1] - current_number
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])
        if not unvisited:
            return 9*number_progress 

        if not self.is_connected(unvisited):
            return float('inf')
        return 9*number_progress + len(unvisited)
    
    def h10(self, pos, visited, current_number):
        number_progress = self.target_sequence[-1] - current_number
        unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited
        ])

        non_numbered_unvisited = set([
            (x, y) for x in range(self.n) for y in range(self.n)
            if (x, y) not in visited and self.board[x][y] is None
        ])

        if not unvisited:
            return number_progress 

        if not self.is_connected(unvisited):
            return float('inf')
        return number_progress + len(non_numbered_unvisited)



    def _a_star(self, start_pos, h):
        # Priority queue stores (f_score, steps_taken, pos, current_number, visited, path)
        heap = []

        # Initial state
        visited = set([start_pos])
        path = [start_pos]
        current_number = self.board[start_pos[0]][start_pos[1]]
        g_score = 0
        f_score = g_score + h(start_pos, visited, current_number)

        heapq.heappush(heap, (f_score, g_score, start_pos, current_number, visited, path))
        self.animation_frames.append((list(path), True))

        while heap:
            #print(len(self.animation_frames))
            f_score, g_score, pos, current_number, visited, path = heapq.heappop(heap)

            if self.board[pos[0]][pos[1]] == self.target_sequence[-1] and len(visited) == self.n * self.n:
                return list(path)

            for dx, dy in DIRS_LIST:
                nx, ny = pos[0] + dx, pos[1] + dy
                next_pos = (nx, ny)

                if not self.in_bounds(nx, ny) or next_pos in visited:
                    continue

                tile_value = self.board[nx][ny]

                # You can only step on 0 or current_number + 1
                if tile_value != 0 and tile_value != current_number + 1:
                    continue

                next_number = current_number + 1 if tile_value == current_number + 1 else current_number

                new_visited = visited.copy()
                new_visited.add(next_pos)
                new_path = path + [next_pos]
                new_g = g_score + 1
                new_f = new_g + h(next_pos, new_visited, next_number)

                heapq.heappush(heap, (new_f, new_g, next_pos, next_number, new_visited, new_path))

                self.animation_frames.append((list(new_path), True))
               

        return None

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

def animate_dfs(board, frames):
    n = len(board)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Draw cell grid (using rectangles)
    for i in range(n):
        for j in range(n):
            rect = plt.Rectangle((j, n - 1 - i), 1, 1, fill=False, edgecolor='gray', linewidth=1, zorder=1)
            ax.add_patch(rect)

    def make_gradient_line(x, y, total_steps):
        """Create segments with a smooth transition from green to purple based on total grid size"""
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Global gradient transition from green to purple based on the total grid length
        colors = [(0, 1, 0), (0.5, 0, 0.5)]  # green to purple (RGB values)
        color_range = np.linspace(0, 1, total_steps)

        # Apply the colors smoothly between green and purple across the entire grid
        segment_colors = [(
            colors[0][0] * (1 - t) + colors[1][0] * t,  # R
            colors[0][1] * (1 - t) + colors[1][1] * t,  # G
            colors[0][2] * (1 - t) + colors[1][2] * t   # B
        ) for t in color_range]
        
        return segments, segment_colors

    # Draw number labels centered in black circles with white, bold text
    for i in range(n):
        for j in range(n):
            val = board[i][j]
            if val != 0:
                # Draw a black circle behind the number with higher zorder
                circle = plt.Circle((j + 0.5, n - 1 - i + 0.5), 0.35, color='black', ec='black', zorder=6)
                ax.add_patch(circle)
                # Draw the number in white, bold text with higher zorder
                ax.text(j + 0.5, n - 1 - i + 0.5, str(val), ha='center', va='center', fontsize=12, color='white', fontweight='bold', zorder=7)

    # Line placeholders (with lower zorder to ensure it's beneath the circles)
    path_collection = LineCollection([], linewidth=25, zorder=2)
    ax.add_collection(path_collection)

    total_repeats = 1

    def update(frame_idx):
        nonlocal total_repeats
        path, is_forward = frames[frame_idx]
        x = [p[1] + 0.5 for p in path]
        y = [n - 1 - p[0] + 0.5 for p in path]

        # Apply the green to purple gradient line across the entire grid
        segments, colors = make_gradient_line(x, y, n * n)
        path_collection.set_segments(segments)
        path_collection.set_color(colors)

        ax.set_title(f"Step {frame_idx + 1} of {len(frames)}, Loop #: {total_repeats}")

        if frame_idx == len(frames)-1:
            total_repeats+=1

        return path_collection,

    ani = FuncAnimation(fig, update, frames=len(frames), interval=300, blit=False, repeat=True)
    plt.show()





def read_board_from_file(filename):
    board = []
    with open(filename, 'r') as file:
        for line in file:
            row = []
            for val in line.strip().split(','):
                try:
                    row.append(int(val))
                except ValueError:
                    row.append(0)
            board.append(row)
    return board

if __name__ == "__main__":
    
    
    filename = sys.argv[1]
    h = None
    if len(sys.argv) > 2:
        h = int(sys.argv[2])

    board = read_board_from_file(filename)
    solver = ZipSolver(board)
    path = solver.solve(h)

    print("\nFinal path found:" if path else "\nNo valid path found.")
    if path:
        print(" -> ".join(map(str, path)))

    animate_dfs(board, solver.animation_frames)
    


    '''
    data = []
    for i in range(1, 51):
        print(i)
        filename = "games/zip"+str(i)+".txt"
        for h in range(1, 11):
            print(h)
            if h == 1:
                h = None
            board = read_board_from_file(filename)
            solver = ZipSolver(board)
            path = solver.solve(h)
            if path == None: print("Couldn't solve ", i, h)
            data.append({"h":h, "size":solver.n, "hints":len(solver.target_sequence), "time":solver.time, "steps":len(solver.animation_frames)})

    print(data)
    '''

    