#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=FDRprompt
CFG=vit_b16
SHOTS=16

DEVICE=$1

for DATASET in imagenet_a imagenet_r imagenet_sketch imagenetv2
do
    for SEED in 1
    do
        DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            CUDA_VISIBLE_DEVICES=${DEVICE} \
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}\
            --load-epoch 200 \
            --eval-only
        fi
    done
done