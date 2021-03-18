#!/bin/bash

MODES=('singlewarp' "multiwarp-5")
ENCODINGS_S=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'monocular')
ENCODINGS_M=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9')

MODEL_DIR="/home/dtejaswi/tensorboard_hpc/artemis_test_b16"

EXPTS=('1' '2' '3' '4' '5' 'm' 'n' 'o' 'p' 'w' 'r' 'wire1' 'wire2' 'wire3')
DATA_ROOT="/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/CheckpointsAndResults/extras/png/teaser_png"

for EXPT in "${EXPTS[@]}"
do
  DATA_DIR="${DATA_ROOT}/${EXPT}"
  SUFFIX="${EXPT}_"

  for MODE in "${MODES[@]}"
  do
    if [ "${MODE}" = "singlewarp" ]; then
      ENCODINGS=( "${ENCODINGS_S[@]}" )
    else
      ENCODINGS=( "${ENCODINGS_M[@]}" )
    fi

    for ENC in "${ENCODINGS[@]}"
    do
      for SEQ in 1
        do
        for EPOCH in 40 60
        do
          echo "${MODEL_DIR}/${MODE}/${ENC}/config.pkl ${SEQ}"
          python3 infer_depth.py --no_pose --sequence_length 2 --config_dir "${MODEL_DIR}/${MODE}/${ENC}/" --seq $SEQ --suffix ${SUFFIX} --data_dir ${DATA_DIR} --use_checkpoint_at $EPOCH
        done
      done
    done
  done
done
