#!/usr/bin/bash

# update params
python3 ../config_create.py
echo "Params updated..."
# train and evaluate models
python3 ../evaluation.py -a y
echo "Models trained and evaluated. Please check output files."
