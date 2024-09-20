#!/bin/bash

python train.py -s data/training-data/train --eval -r 8 --iterations 7000 --densify_grad_threshold 0.00002 --sh_degree 3

# python train.py -s data/training-data/train --eval -r 8 --iterations 7000
