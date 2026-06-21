#!/bin/bash
set -e

python3 ./main.py BFS_PI_PoCL_config.json 
python3 ./main.py FFT_PI_PoCL_config.json
python3 ./main.py KMeans_Pi_PoCL_config.json
python3 ./main.py SPMV_PI_PoCL_config.json
python3 ./main.py SRAD_PI_PoCL_config.json

