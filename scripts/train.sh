#!/bin/bash

cd app

# Runs python in the background, redirects stdout and stderr to logs/train.log
nohup python main.py train_vj -cpus 12 -ff -mf 15000 -tpr 0.99 &> ../logs/train_99.log &
