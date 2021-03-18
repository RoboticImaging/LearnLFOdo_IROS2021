#!/bin/bash

# This scrips runs rpe

#charizard
cd /media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/code/tum_poses || exit

RUN="artemis_test_b16_interarea_tv_mean"
SEQ="82"

MODES=('singlewarp' "multiwarp-5")
ENCODINGS_S=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'monocular' 'epi_without_disp_stack')
ENCODINGS_M=('epi' 'stack' 'focalstack-17-5' 'focalstack-17-9' 'epi_without_disp_stack')

REF_FILE="tum_gt_first_cam_${SEQ}.txt"
RPE_TRANS_OUT_FILE="${RUN}/${SEQ}_rpe_translation_evaluation_na.txt"
RPE_ROT_OUT_FILE="${RUN}/${SEQ}_rpe_rotation_evaluation_na.txt"

if [ -f "${RPE_OUT_FILE}" ]; then
  echo "${RPE_OUT_FILE} exists."
else

  for MODE in "${MODES[@]}"
  do
    if [ "${MODE}" = "singlewarp" ]; then
      ENCODINGS=( "${ENCODINGS_S[@]}" )
    else
      ENCODINGS=( "${ENCODINGS_M[@]}" )
    fi

    for ENC in "${ENCODINGS[@]}"
    do
      EST_FILE="${RUN}/${MODE}_${ENC}_${SEQ}_tum_est_first_cam.txt"
      echo "${MODE} ${ENC} ${SEQ}" >> ${RPE_TRANS_OUT_FILE}
      echo "${MODE} ${ENC} ${SEQ}" >> ${RPE_ROT_OUT_FILE}

      evo_rpe tum ${REF_FILE} ${EST_FILE} --pose_relation trans_part --delta 1 --delta_unit f >> ${RPE_TRANS_OUT_FILE}
      evo_rpe tum ${REF_FILE} ${EST_FILE} --pose_relation angle_deg --delta 1 --delta_unit f >> ${RPE_ROT_OUT_FILE}
    done
  done
fi