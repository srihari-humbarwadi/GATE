#!/bin/bash
# ----------------------------------------------------------
# Install mamba
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
# Fix the issue wth the setup, autopath etc.
bash $HOME/mamba.sh -bf -p $HOME/environment_management/mambaforge/
# ----------------------------------------------------------
# Create a python3.8 environment in mamba
mamba create -n gate-env python=3.8
mamba activate gate-env
# ----------------------------------------------------------
# Install GATE dependencies
mamba install git -y
mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
mamba install opencv -y
mamba install h5py -y
mamba install pytorch-lightning -y
mamba install transformers -y
mamba install GPUtil -y
mamba install orjson -y
mamba install tqdm -y
mamba install regex -y
mamba install pudb -y
mamba install seaborn -y
mamba install scikit-learn -y
mamba install pytest -y
mamba install rich -y
mamba install python-dotenv -y
mamba install flake8 -y
mamba install isort -y
mamba install black -y
mamba install pre-commit -y
mamba install wandb -y
mamba install hydra-core -y
mamba install google-cloud-storage -y
mamba install google-api-python-client -y
mamba install cryptography -y
mamba install ftfy -y
mamba install imutils -y
mamba install scipy -y
mamba install einops -y
mamba install torchmetrics -y
mamba install ffmpeg -y
echo yes | pip install hub timm jsonlint nvidia-ml-py3 testresources hydra hydra-core hydra-colorlog hub hydra-optuna-sweeper dotted_dict ray higher --upgrade
echo yes | pip install git+https://github.com/openai/CLIP.git@main
echo yes | pip install git+https://github.com/AntreasAntoniou/TALI.git@main
echo yes | pip install git+https://github.com/tensorflow/datasets.git@master
echo yes | pip install tensorflow_datasets
echo yes | pip install tf-nightly
echo yes | pip install learn2learn
#git clone https://github.com/AntreasAntoniou/GATE.git#prototypical_network_integration

echo yes | pip install -e .
# ----------------------------------------------------------
# Install development dependencies
mamba install bash sh htop jupyterlab -y
mamba install micro bat -y
mamba install -c mamba-forge git-lfs -y

mamba install black -y
mamba install starship tmux -y
mamba install micro bat -y
echo yes | pip install nvitop
echo yes | pip install pytest-pretty-terminal
echo yes | pip install glances
echo yes | pip install loguru
