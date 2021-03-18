python3 train_multiwarp_learn_weights.py focalstack \
/project/CIAutoInterp/data/module-1-1/module1-2-png \
artemis_test_b16_interarea_learn/multiwarp-5/focalstack-17-9 \
--save-path /project/CIAutoInterp/results \
--num-cameras 17 \
--num-planes 9 \
-c  3 8 13 7 9 \
--sequence-length 2 \
-b16 -s0.3 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-j16
