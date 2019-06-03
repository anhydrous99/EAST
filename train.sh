#!/bin/sh

python train.py --input_size=512 --batch_size=6 --nb_workers=6 --training_data_path=ICDAR2013+2015/train_data/ --validation_data_path=valid/ --checkpoint_path=tmp/icdar2015_east_mobilenetv2/
