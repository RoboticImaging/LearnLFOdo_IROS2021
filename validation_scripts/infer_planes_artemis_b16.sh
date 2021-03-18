#!/bin/bash

#MODES=('singlewarp' "multiwarp-5")
#MODES=("multiwarp-outer")
#MODES=("multiwarp-outer")
MODES=("multiwarp-5")

ENCODINGS_S=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'monocular' 'epi_without_disp_stack')
#ENCODINGS_M=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'epi_without_disp_stack')
#ENCODINGS_M=('stack' 'focalstack-17-5' 'focalstack-17-9' 'epi_without_disp_stack')
ENCODINGS_M=('epi_without_disp_stack')

# for old planes
#MODEL_DIR="/home/dtejaswi/tensorboard_hpc/artemis_test_b16"
#EXPTS=('f' 'b' 'dgd')
#PLANES=('20' '40' '60' '80' '100')
#DATA_ROOT="/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/CheckpointsAndResults/extras/png/teaser_png"

# for new planes
#MODEL_DIR="/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea"
#EXPTS=('planes')
#PLANES=('400' '425' '450' '475' '500' '525' '550' '575' '600' '625' '650' '675' '700' '725' '750' '775' '800')
#DATA_ROOT="/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png"

# for new planes with tex
MODEL_DIR="/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea"
EXPTS=('planes_tex')
PLANES=('400' '500' '600' '700' '800')
DATA_ROOT="/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png"

for EXPT in "${EXPTS[@]}"
do
  for PLANE in "${PLANES[@]}"
  do
    DATA_DIR="${DATA_ROOT}/${EXPT}/${PLANE}"
    SUFFIX="plane_${EXPT}_${PLANE}_"

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
          for EPOCH in 40
          do
            echo "${MODEL_DIR}/${MODE}/${ENC}/config.pkl ${SEQ}"
            python3 infer_depth.py --no_pose --sequence_length 2 --config_dir "${MODEL_DIR}/${MODE}/${ENC}/" --seq $SEQ --suffix ${SUFFIX} --data_dir ${DATA_DIR} --use_checkpoint_at $EPOCH
          done  # epochs
        done  # sequence
      done  # encoding
    done  # modes
  done  # planes
done  # experiments
