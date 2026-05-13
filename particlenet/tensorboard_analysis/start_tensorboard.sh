#!/bin/bash
# Start TensorBoard to visualize training runs.
# Usage: ./start_tensorboard.sh [PORT]
#   PORT defaults to 6006 if not specified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$(dirname "$SCRIPT_DIR")/.venv"
LOG_DIR="$SCRIPT_DIR/runs"
PORT="${1:-6006}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: virtual environment not found at $VENV_DIR"
    exit 1
fi

echo "Starting TensorBoard on port $PORT..."
echo "Log directory: $LOG_DIR"
echo "Open http://localhost:$PORT in your browser."
echo "Press Ctrl+C to stop."

"$VENV_DIR/bin/tensorboard" --logdir="$LOG_DIR" --port "$PORT" --bind_all
