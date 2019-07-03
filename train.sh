#!/bin/sh

python train.py --input_size=512 --batch_size=6 --nb_workers=2 --training_data_path=data/train_data/ --validation_data_path=data/validation_data/ --checkpoint_path=tmp/cocotext_icdar2013_15_east_mobilenet_v2_11_06_2019/
