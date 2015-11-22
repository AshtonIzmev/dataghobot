#!/bin/bash

if [ -d data ]; then
    echo "data directory already present, exiting"
    exit 1
fi

mkdir data
wget --recursive --level=1 --cut-dirs=3 --no-host-directories --directory-prefix=data \
 http://archive.ics.uci.edu/ml/machine-learning-databases/adult/


