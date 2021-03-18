python3 train_multiwarp.py epi \
/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-1-png \
test_multi \
-c 3 8 13 7 9 \
--save-path /media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/checkpoints \
--sequence-length 2 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray \
-e full \
-j2
