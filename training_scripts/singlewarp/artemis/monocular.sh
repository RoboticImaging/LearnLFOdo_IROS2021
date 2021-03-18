python3 train_multiwarp.py stack \
/project/CIAutoInterp/data/module-1-1/module1-1-png \
artemis_test/singlewarp/monocular \
--save-path /project/CIAutoInterp/results \
-c 8 \
-k input \
--sequence-length 2 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray \
-j16
