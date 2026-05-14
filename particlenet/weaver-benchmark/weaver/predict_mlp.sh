#!/bin/bash

# The environment
python3 -m venv weaver_env_p100
source weaver_env_p100/bin/activate

#Make sure all packages are installed: the installation should be skipped if they exist
# pip install weaver-core
# pip install torch
# pip install uproot3

PREFIX='mlp'
MODEL_CONFIG='mlp_pf.py'
DATA_CONFIG='pf_features.yaml'
PATH_TO_SAMPLES='top_tagging/samples/'

python train.py --predict \
 --data-test ${PATH_TO_SAMPLES}'/prep/top_test_*.root' \
 --num-workers 3 \
 --data-config top_tagging/data/${DATA_CONFIG} \
 --network-config top_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX}_best_epoch_state.pt \
 --gpus 0 --batch-size 1024 \
 --predict-output output/${PREFIX}_predict.root
