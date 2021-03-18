python3 train_multiwarp.py epi \
/project/CIAutoInterp/data/module-1-1/module1-1-png \
artemis_test/multiwarp-5/epi/ \
-c 3 8 13 7 9 \
--save-path /project/CIAutoInterp/results \
--sequence-length 2 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray \
-e full \
-j16