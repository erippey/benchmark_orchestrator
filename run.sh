#!/bin/bash
set -e

#python3 ./main.py BFS_OPI_OpenCL_config.json 
#python3 ./main.py FFT_OPI_OpenCL_config.json 
#python3 ./main.py KMeans_OPI_OpenCL_config.json
python3 ./main.py SPMV_OPI_OpenCL_config.json
#python3 ./main.py SRAD_OPI_OpenCL_config.json

