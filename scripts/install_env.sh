#!/bin/bash

conda create --name BIOM_env python=3.13 -y
conda activate BIOM_env

# Install uv global on you pc or only on this env
# if only on this env:
pip install uv
# if global, no need to install anything extra, just:
uv pip install -r requirements.txt

pip install -e .

uv pip install ipykernel
python -m ipykernel install --user --name=BIOM_env --display-name "BIOM_env (Conda)"