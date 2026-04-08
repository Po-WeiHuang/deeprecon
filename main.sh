#!/bin/bash
export WANDB_CACHE_DIR="/data/snoplus2/weiiiii/uv/deeprecon/.wandb_cache"
#export PYTORCH_ALLOC_CONF="expandable_segments:True"
mkdir -p $WANDB_CACHE_DIR
export WANDB_API_KEY="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"
source /home/huangp/deepsno/.venv/bin/activate
python /home/huangp/deeprecon/main.py