#!/bin/bash

# The environment
python3 -m venv weaver_env_p100
source weaver_env_p100/bin/activate

#Make sure all packages are installed: the installation should be skipped if they exist
# pip install weaver-core
# pip install torch
# pip install uproot3

PREFIX='deepak8'
MODEL_CONFIG='deepak8_pf.py'
DATA_CONFIG='pf_features.yaml'
PATH_TO_SAMPLES='top_tagging/samples/'

python train.py \
 --data-train ${PATH_TO_SAMPLES}'/prep/top_train_*.root' \
 --data-val ${PATH_TO_SAMPLES}'/prep/top_val_*.root' \
 --fetch-by-file --fetch-step 1 --num-workers 3 \
 --data-config top_tagging/data/${DATA_CONFIG} \
 --network-config top_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX} \
 --gpu '' --batch-size 1024 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 --log output/${PREFIX}.train.log
