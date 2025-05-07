import os

# Specify the folder containing the files
folder_path = './tangoboards/'  # Replace with your folder path

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter out non-file items (directories, etc.)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

# Sort the files (this step can be skipped if order doesn't matter)
files.sort()

# Rename each file
for i, file in enumerate(files, 1):
    # Construct the new file name
    new_name = f"tango{i}.txt"
    
    # Get the full paths
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print(f"Renamed {len(files)} files.")
