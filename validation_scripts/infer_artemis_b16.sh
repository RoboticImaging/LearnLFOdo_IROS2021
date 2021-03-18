#!/bin/bash

MODES=('singlewarp' "multiwarp-5")
#MODES=("multiwarp-all")
ENCODINGS_S=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'monocular' 'epi_without_disp_stack')
ENCODINGS_M=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'epi_without_disp_stack')
#ENCODINGS_M=('stack' 'focalstack-17-5' 'focalstack-17-9' 'epi_without_disp_stack')

#ENCODINGS_S=('epi_without_disp_stack')
#ENCODINGS_M=('epi_without_disp_stack')

MODEL_DIR="/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv_mean"
SUFFIX="seq"
DATA_DIR="/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png/"
CHECKPOINT=40

for MODE in "${MODES[@]}"
do
  if [ "${MODE}" = "singlewarp" ]; then
    ENCODINGS=( "${ENCODINGS_S[@]}" )
  else
    ENCODINGS=( "${ENCODINGS_M[@]}" )
  fi

  for ENC in "${ENCODINGS[@]}"
  do
#    for SEQ in 16 44 50 51 52    # validation sequences for linear interpolation
#    for SEQ in 16 28 40 44       # validation sequences for area interpolation - using this now
#    for SEQ in 16 28 40 44 60 61 62 80 81 82 83          # test sequences
#    for SEQ in 80 81 82 83           # new test sequences
    for SEQ in 12
    do
      echo "${MODEL_DIR}/${MODE}/${ENC}/config.pkl ${SEQ}"
      python3 infer_multiwarp.py --sequence_length 2 --config_dir "${MODEL_DIR}/${MODE}/${ENC}/" --seq $SEQ --suffix ${SUFFIX} --data_dir ${DATA_DIR} --use_checkpoint_at ${CHECKPOINT}
    done
  done
done

#
#for SEQ in 16 50 51 52
#do
#  python3 infer_multiwarp.py --config "${DIR_NAME}//singlewarp/monocular/config.pkl" --seq $SEQ
##  echo "${DIR_NAME}/singlewarp/monocular/config.pkl ${SEQ}"
#done