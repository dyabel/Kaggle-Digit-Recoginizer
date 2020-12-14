#!/bin/bash

python='python3'

CUDA_VISIBLE_DEVICES=$1 ${python} main.py --net 0 --all_data 0 --config_path cnn_config.json
