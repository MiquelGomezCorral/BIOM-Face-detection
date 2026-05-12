#!/bin/bash

cd app

# Runs python in the background, redirects stdout and stderr to logs/train.log
nohup python main.py train_vj -cpus 12 -ff -rt -mf 22000 -tpr 0.999 -pfp &> ../logs/train.log &