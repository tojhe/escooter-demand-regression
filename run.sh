#!/usr/bin/bash
# run preprocessing
python3 ./mlp/preprocessing.py -ts 0.2 -l # <address of blob - redacted for privacy> 
echo "Preprocessing completed..."
# update params
python3 ./mlp/config_create.py
echo "Params updated..."
# train and evaluate models
python3 ./mlp/evaluation.py -a y
echo "Models trained and evaluated. Please check output files."
