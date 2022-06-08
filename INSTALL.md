1. Navigate to the directory in which you wish to install the mamba package manager, and download the relevant miniforge installation:
    ```bash
    wget https://github.com/mamba-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O $HOME/mamba.sh
    ```
2. Install miniforge:
    ```bash 
    bash $HOME/mamba.sh -bf -p $HOME/mamba/
    ```
3. Create a mamba environment variable and add it to your bash profile. Then source it into the shell:
    ```bash 
    mamba_DIR=$HOME/mamba/
    source $mamba_DIR/bin/activate
    ``` 
4. [Optional] If you prefer to have the mamba variable loaded upon opening your terminal, and your environment sourced, add the new mamba environment variable to your ~/.bashrc file, along with the source command
   ```bash
   echo "export 'mamba_DIR=${mamba_DIR}'" >> $HOME/.bashrc
   echo "source $mamba_DIR/bin/activate" >> $HOME/.bashrc # add this to your .bashrc 
   ```
5. Create a mamba environment for the GATE framework:
    ```bash
    mamba create -n GATE-env python=3.8 -y
    mamba activate GATE-env
    ```
6. Install pytorch and torchvision:
    ```bash
   mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly -y
   mamba install opencv -y
   mamba install h5py -y
   mamba install pytorch-lightning -y
   mamba install transformers -y
   mamba install GPUtil -y
   mamba install orjson -y
   mamba install tqdm -y
   mamba install regex -y
   mamba install pudb -y
   mamba install jupyterlab -y
   mamba install seaborn -y
   mamba install scikit-learn -y
   mamba install bash sh -y
   mamba install pytest -y
   mamba install rich -y
   mamba install python-dotenv flake8 isort black pre-commit wandb hydra-core google-cloud-storage google-api-python-client cryptography ftfy imutils scipy einops torchmetrics ffmpeg htop nvtop -y
   pip install jsonlint nvidia-ml-py3 testresources hydra hydra-colorlog hub hydra-optuna-sweeper dotted_dict
   pip install git+https://github.com/activeloopai/Hub.git@skip/agreement
   or pip install hub
   pip install git+https://github.com/openai/CLIP.git@main
   pip install git+https://github.com/AntreasAntoniou/TALI.git@main
   ```
7. Install the GATE framework:
    ```bash
    export CODE_DIR=$HOME/GATE
    echo "export 'CODE_DIR=${CODE_DIR}'" >> $HOME/.bashrc
    cd $HOME
    git clone https://github.com/AntreasAntoniou/GATE.git $CODE_DIR
    cd $CODE_DIR
    
    pip install -r $CODE_DIR/requirements.txt
    pip install -e $CODE_DIR
   ```
8. Install fancy-pants dev environment tools:
   ```bash
   mamba install -c mamba-forge git-lfs -y
   mamba install starship tmux -y
   mamba install gh --channel mamba-forge -y
   mamba install htop nvtop -y
   mamba install ffmpeg libsm6 libxext6  -y
   mamba install bat micro -y
   ```