import subprocess

commands = [
    ["python3", "Zip/Zip.py", "Zip/games/zip8.txt"],
    ["python3", "Zip/Zip.py", "Zip/games/zip8.txt", "2"],
    ["python3", "Zip/Zip.py", "Zip/games/zip8.txt", "3"],
    ["python3", "Queens/Queens.py", "Queens/games/queens4.txt", "0"],
    ["python3", "Queens/Queens.py", "Queens/games/queens4.txt", "1"],
    ["python3", "Queens/Queens.py", "Queens/games/queens4.txt", "2"],
    ["python3", "Tango/Tango.py", "Tango/games/tango58.txt", "0"],
    ["python3", "Tango/Tango.py", "Tango/games/tango58.txt", "1"],
]

processes = []

# Start all processes
for cmd in commands:
    print(f"Launching: {' '.join(cmd)}")
    # Let Queens.py run with GUI (no output capture)
    if "Queens.py" in cmd[1]:
        proc = subprocess.Popen(cmd)
        processes.append((cmd, proc, False))  # False = no output capture
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((cmd, proc, True))  # True = capture output

# Collect output for those with output capture
for cmd, proc, should_capture in processes:
    if should_capture:
        stdout, stderr = proc.communicate()
        print(f"\nFinished: {' '.join(cmd)}")
        print("Output:")
        print(stdout)
        if stderr:
            print("Errors:")
            print(stderr)
        print("=" * 40)
    else:
        proc.wait()
        print(f"\nFinished: {' '.join(cmd)} (GUI likely shown)")
        print("=" * 40)
