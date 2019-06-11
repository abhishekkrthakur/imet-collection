#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="se_resnext101_32x4d"
export TRAINING_BATCH_SIZE=64
export TEST_BATCH_SIZE=32
export NUM_CLASSES=1103
export IMAGE_SIZE=320
export EPOCHS=25
python3 train.py --fold 0
python3 predict.py --fold 0
python3 train.py --fold 1
python3 predict.py --fold 1
python3 train.py --fold 2
python3 predict.py --fold 2
python3 train.py --fold 3
python3 predict.py --fold 3
python3 train.py --fold 4
python3 predict.py --fold 4
