python3 train_multiwarp.py focalstack \
/project/CIAutoInterp/data/module-1-1/module1-1-png \
artemis_test_b16/singlewarp/focalstack-17-5 \
--save-path /project/CIAutoInterp/results \
--num-cameras 17 \
--num-planes 5 \
-c 8 \
--sequence-length 2 \
-b16 -s0.3 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-j16
