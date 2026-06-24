#!/bin/bash
set -e


python3 ./main_nano.py SPMV_nano_clvk_config.json -o
python3 ./main_nano.py SRAD_nano_clvk_config.json -o
python3 ./main_nano.py BFS_nano_clvk_config.json
python3 ./main_nano.py FFT_nano_clvk_config.json
python3 ./main_nano.py KMeans_nano_clvk_config.json
python3 ./main_nano.py SPMV_nano_clvk_config.json
python3 ./main_nano.py SRAD_nano_clvk_config.json