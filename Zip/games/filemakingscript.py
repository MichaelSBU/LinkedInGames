import os
import sys
from bs4 import BeautifulSoup

def extract_grid_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Create a grid based on the positions
    grid = {}

    for cell in soup.find_all('div', class_='game-cell'):
        pos = cell.get('data-pos')
        if pos:
            row, col = map(int, pos.split(','))
            inner = cell.find('div', class_='number-circle')
            val = int(inner.get_text(strip=True)) if inner and inner.get_text(strip=True).isdigit() else 0
            grid[(row, col)] = val

    # Find grid dimensions
    max_row = max(r for r, _ in grid.keys())
    max_col = max(c for _, c in grid.keys())

    # Convert grid dict to 2D list
    matrix = [
        [grid.get((r, c), 0) for c in range(max_col + 1)]
        for r in range(max_row + 1)
    ]

    return matrix

def save_matrix_to_file(matrix, file_path):
    with open(file_path, "w") as f:
        for row in matrix:
            # Replace 0s with empty string
            row_strs = [str(cell) if cell != 0 else "" for cell in row]
            f.write(",".join(row_strs) + "\n")

# Main process
html_file_paths = ["zip"+str(i)+".txt" for i in range(6, 53)]

for html_file_path in html_file_paths:
    grid_matrix = extract_grid_from_html(html_file_path)

    # Optionally, print the matrix
    for row in grid_matrix:
        print(row)
    print()

    # Save the result to the same file or another location
    save_matrix_to_file(grid_matrix, html_file_path)
