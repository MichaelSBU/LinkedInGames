import os
import ast

# Modify this path to the directory containing your files
folder_path = "/Users/mikeybudney/Desktop/CSE537/Project/Queens/games/convert"

def convert_rgb_files_to_labels(folder):
    for filename in os.listdir(folder):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(folder, filename)
        with open(file_path, "r") as f:
            data = ast.literal_eval(f.read())  # Safely parse the list

        # Flatten and get unique colors
        unique_colors = {}
        label = 0
        for row in data:
            for cell in row:
                if cell not in unique_colors:
                    unique_colors[cell] = label
                    label += 1

        # Convert to labeled format
        converted = [[unique_colors[cell] for cell in row] for row in data]

        # Write back in simple integer CSV format
        with open(file_path, "w") as f:
            for row in converted:
                f.write(",".join(map(str, row)) + "\n")

# Run the conversion
convert_rgb_files_to_labels(folder_path)
