#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=FDRprompt

GPU=$1
CFG=vit_b16_ep20_batch4
SHOTS=16

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars ucf101 sun397
    do
    for SEED in  1 2 3
        do
            DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
            if [ -d "$DIR" ]; then
                echo "Results are available in ${DIR}."
            else
                echo "Run this job and save the output to ${DIR}"
                CUDA_VISIBLE_DEVICES=${GPU} python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES base
            fi
      done
done