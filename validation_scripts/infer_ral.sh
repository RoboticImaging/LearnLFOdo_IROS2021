#!/bin/bash

#MODES=('multiwarp-5')
#ENCODINGS=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9')
MODES=('singlewarp' "multiwarp-5")
ENCODINGS_S=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'monocular')
ENCODINGS_M=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9')

MODEL_DIR="/home/dtejaswi/Desktop/joseph_daniel/ral"
SUFFIX="wire3_"
#DATA_DIR="/home/dtejaswi/Desktop/joseph_daniel/extras/png/B/20"
DATA_DIR="/home/dtejaswi/Desktop/joseph_daniel/extras/png/teaser_png/wire/3/"


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
      echo "${MODEL_DIR}/${MODE}/${ENC}/config.pkl ${SEQ}"
      python3 infer_depth.py --config "${MODEL_DIR}/${MODE}/${ENC}/" --seq $SEQ --suffix ${SUFFIX} --data_dir ${DATA_DIR} --no_pose
    done
  done
done

#
#for SEQ in 16 50 51 52
#do
#  python3 infer_multiwarp.py --config "${DIR_NAME}//singlewarp/monocular/config.pkl" --seq $SEQ
##  echo "${DIR_NAME}/singlewarp/monocular/config.pkl ${SEQ}"
#done