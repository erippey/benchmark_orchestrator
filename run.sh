#!/bin/bash
set -e

python3 ./main.py FFT_PI_CPU_config.json
python3 ./main.py FFT_PI_GPU_config.json
python3 ./main.py KMeans_PI_GPU_config.json 
python3 ./main.py KMeans_PI_CPU_config.json 
python3 ./main.py KMeans_PI_OMP_config.json 
python3 ./main.py SRAD_PI_CPU_config.json 
python3 ./main.py SRAD_PI_GPU_config.json 