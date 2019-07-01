#!/bin/bash

GPU_ID=1
BATCH_SIZE=8
WORKER_NUMBER=32
LEARNING_RATE=0.001
DECAY_STEP=4
EPOCHS=12

SESSION=1
EPOCH=6
CHECKPOINT=79230

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset ucf24 --net p3d199 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --epochs $EPOCHS --cag --cuda \
				   --r True  --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT
				   