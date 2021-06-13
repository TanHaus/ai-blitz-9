# ai-blitz-9
AI Blitz #9 - Hello NLP

## Install

Dependencies

- pytorch
- transformers (for all problems except sound prediction)
- nemo (for sound prediction)

Not required to run model

- numpy, pandas matplotlib
- tensorboard (for logging)
- jiwer (for evaluation)

Tested environment: Ubuntu (Linux), python 3.8, torch 1.8.1, cuda 11.1

```bash
conda create -n ai-blitz python=3.8
conda activate ai-blitz

conda install torch cudatoolkit=11.1 -c pytorch -c nvidia
conda install numpy, pandas, matplotlib, tensorboard

pip install transformers, jiwer
pip install git+https://github.com/NVIDIA/NeMo.git#egg=nemo_toolkit[all]
```

## Download datasets

Go to the AI Blitz #9 page to download the datasets: https://www.aicrowd.com/challenges/ai-blitz-9

You can also download the dataset via `aicrowd-cli`

```bash
pip install aicrowd-cli
aicrowd login

aicrowd dataset download --challenge sound-prediction -j 3 -o data
```

After downloading the datasets, extract and place them under `data/` directory under each problem directory.
