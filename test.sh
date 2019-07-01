#!/bin/bash

GPU_ID=1
SESSION=1
EPOCH=7
CHECKPOINT=79230

CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset ucf24 --net p3d199 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cag --cuda
				   
