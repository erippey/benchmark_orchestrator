#!/bin/bash
set -e

#python3 main.py

git add .

git commit -m "more logging data $(date)"

git push origin main