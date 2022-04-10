1. Navigate to the directory in which you wish to install the conda package manager, and download the relevant miniforge installation:
    ```bash
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O $HOME/conda.sh
    ```
2. Install miniforge:
    ```bash 
    bash $HOME/conda.sh -bf -p $HOME/conda/
    ```
3. Create a conda environment variable and add it to your bash profile. Then source it into the shell:
    ```bash 
    CONDA_DIR=$HOME/conda/
    source $CONDA_DIR/bin/activate
    ``` 
4. [Optional] If you prefer to have the conda variable loaded upon opening your terminal, and your environment sourced, add the new conda environment variable to your ~/.bashrc file, along with the source command
    ```bash
   echo "export 'CONDA_DIR=${CONDA_DIR}'" >> $HOME/.bashrc
   echo "source $CONDA_DIR/bin/activate" >> $HOME/.bashrc # add this to your .bashrc 
    ```
5. Create a conda environment for the GATE framework:
    ```bash
    conda create -n GATE-env python=3.8 -y
    conda activate GATE-env
    ```
6. Install pytorch and torchvision:
    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly -y
   conda install opencv -y
   conda install h5py -y
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