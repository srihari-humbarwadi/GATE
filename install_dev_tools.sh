#!/bin/bash
# Install development dependencies
mamba install bash htop jupyterlab -y
mamba install micro bat -y
mamba install -c mamba-forge gh git-lfs -y
mamba install black -y
mamba install starship tmux -y

echo yes | pip install nvitop
echo yes | pip install pytest-pretty-terminal
echo yes | pip install glances
echo yes | pip install loguru
