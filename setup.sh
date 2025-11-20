# Check if 'uv' is installed, else install it
if ! command -v uv &> /dev/null; then
    echo "[setup.sh] 'uv' not found. Installing with pip..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    echo "export PATH=\"$HOME/.local/bin:$PATH\"" >> ~/.profile
    echo "source $HOME/.local/bin/env" >> ~/.profile
    source ~/.profile
else
    echo "[setup.sh] 'uv' is already installed."
fi

uv sync
uv add flash-attn --no-build-isolation