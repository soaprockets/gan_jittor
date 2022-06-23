#!/usr/bin/bash
python train.py --eval --n_epochs 100 --batch_size 64 --lr 0.0002 --b1 0.5 --b2 0.999 --n_cpu 8 --latent_dim 100 --n_classes 10 --img_size 32 --channels 1 --sample_interval 1000