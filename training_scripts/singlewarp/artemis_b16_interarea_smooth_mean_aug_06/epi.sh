python3 train_multiwarp.py epi \
/project/CIAutoInterp/data/module-1-1/module1-3-png \
artemis_b16_interarea_smooth_mean_aug_06/singlewarp/epi/ \
-c 8 \
--save-path /project/CIAutoInterp/results \
--sequence-length 2 \
-b16 -s0.6 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-e full \
-j16