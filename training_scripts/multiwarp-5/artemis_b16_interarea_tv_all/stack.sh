python3 train_multiwarp.py stack \
/project/CIAutoInterp/data/module-1-1/module1-2-png \
artemis_test_b16_interarea_tv/multiwarp-all/stack \
--save-path /project/CIAutoInterp/results \
-c 1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 \
-k full \
--sequence-length 2 \
-b16 -s0.3 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 100 \
--log-output \
--gray \
-j16
