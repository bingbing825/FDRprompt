#!/bin/bash

# custom config
DATA=/path/to/datasets
TRAINER=FDRprompt

GPU=$1
SUB=$2
SHOTS=16
CFG=vit_b16_ep20_batch4
LOADEP=20

for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars ucf101 sun397
    do
    for SEED in  1 2 3
        do
            COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
            MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
            DIR=output/test_${SUB}/${COMMON_DIR}
            if [ -d "$DIR" ]; then
                echo "Evaluating model"
                echo "Results are available in ${DIR}."
            else
                CUDA_VISIBLE_DEVICES=${GPU} python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                DATASET.NUM_SHOTS ${SHOTS} \
                DATASET.SUBSAMPLE_CLASSES ${SUB}
            fi
      done
done