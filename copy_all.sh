#!/bin/bash

cd "$1"


for d in */; do
    [ "$d" = "run1/" ] && continue
    cp run1/config_metadata.txt "$d"
done