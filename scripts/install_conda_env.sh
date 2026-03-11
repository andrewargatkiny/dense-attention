#!/bin/bash

conda create -y -n ar python=3.13
conda activate ar

for package in $(cat requirements.txt);
do
    conda install -y "$package"
done

pip install torch clearml deepspeed wandb
