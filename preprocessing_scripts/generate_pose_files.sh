#!/bin/bash

INPUT_DIR="/media/dtejaswi/Seagate Expansion Drive/JoeDanielThesisData/data/sequences"
SUFFIX="seq"

for SEQ in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27 28 29 30 31 32 33 34 35 36 37 38 39 40 42 43 43 44 45
#for SEQ in 2 3 4
do
  python3 preprocess/process_poses_correctly.py "${INPUT_DIR}/${SUFFIX}${SEQ}"
done