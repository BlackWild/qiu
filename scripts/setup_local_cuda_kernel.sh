#!/usr/bin/env bash
set -e

# ------------------------------
# CONFIGURATION
# ------------------------------

# Name for the local kernel
KERNEL_NAME="project-cuda"

# ------------------------------
# Detect local .venv created by uv
# ------------------------------
VENV_DIR=".venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ No .venv directory found. Create it with:"
    echo "   uv venv"
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Python not found in .venv. Something is wrong."
    exit 1
fi

echo "✔ Using Python from: $VENV_PYTHON"

# ------------------------------
# Determine LD_LIBRARY_PATH
# ------------------------------
if [ -z "$LD_LIBRARY_PATH" ]; then
    echo "⚠ LD_LIBRARY_PATH is empty. Make sure CUDA libraries are loaded."
    CUDA_ENV_PATH=""
else
    CUDA_ENV_PATH="$LD_LIBRARY_PATH"
fi

echo "✔ Using LD_LIBRARY_PATH: $CUDA_ENV_PATH"

# ------------------------------
# Create local kernel directory inside .venv
# ------------------------------
KERNEL_DIR="$VENV_DIR/share/jupyter/kernels/$KERNEL_NAME"

mkdir -p "$KERNEL_DIR"

KERNEL_JSON="$KERNEL_DIR/kernel.json"

echo "🧱 Creating kernel JSON at $KERNEL_JSON"

# ------------------------------
# Write kernel.json
# ------------------------------
cat > "$KERNEL_JSON" <<EOL
{
  "argv": [
    "$VENV_PYTHON",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python ($KERNEL_NAME)",
  "language": "python",
  "env": {
    "LD_LIBRARY_PATH": "$CUDA_ENV_PATH"
  }
}
EOL

echo "✔ Kernel JSON created."

# ------------------------------
# Instructions for VSCode
# ------------------------------
echo ""
echo "🎉 Setup complete!"
echo "To use this kernel in VSCode Interactive Window or notebooks:"
echo "1. Open Command Palette → 'Jupyter: Select Kernel'"
echo "2. Select the kernel located at:"
echo "   $KERNEL_JSON"
echo ""
echo "This kernel is fully local to your project and works with uv .venv."
