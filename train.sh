#!/bin/sh

python train.py --gpu_list 0,1 --input_size=512 --batch_size=24 --nb_workers=10 --training_data_path=data/train_data/ --validation_data_path=data/validation_data/ --checkpoint_path=tmp/cocotext_icdar2013_15_east_mobilenet_v2_11_06_2019/
