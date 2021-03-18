python3 train_multiwarp.py focalstack \
/project/CIAutoInterp/data/module-1-1/module1-1-png \
artemis_test/singlewarp/focalstack-17-9 \
--save-path /project/CIAutoInterp/results \
--num-cameras 17 \
--num-planes 9 \
-c 8 \
--sequence-length 2 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray \
-j16
