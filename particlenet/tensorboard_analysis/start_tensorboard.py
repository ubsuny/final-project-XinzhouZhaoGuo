"""
Start TensorBoard to visualize training runs.
Run this file directly in VS Code (click the Run button).

NOTE (WSL2): The URL printed by TensorBoard may show a .localdomain address
that Windows browsers cannot resolve. Use the URLs printed below instead.
"""

import os
import socket
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), ".venv")
LOG_DIR = os.path.join(SCRIPT_DIR, "runs")
PORT = 6006

tensorboard_bin = os.path.join(VENV_DIR, "bin", "tensorboard")

if not os.path.exists(tensorboard_bin):
    print(f"Error: tensorboard not found at {tensorboard_bin}")
    sys.exit(1)

# Get WSL2 IP for use in Windows browser
try:
    wsl_ip = socket.gethostbyname(socket.gethostname())
except Exception:
    wsl_ip = None

print(f"Starting TensorBoard on port {PORT}...")
print(f"Log directory: {LOG_DIR}")
print()
print("Open one of these URLs in your Windows browser:")
print(f"  http://localhost:{PORT}        (try this first)")
if wsl_ip:
    print(f"  http://{wsl_ip}:{PORT}  (use this if localhost doesn't work)")
print()
print("Press Ctrl+C to stop.\n")

subprocess.run([tensorboard_bin, "--logdir", LOG_DIR, "--port", str(PORT), "--bind_all"])
