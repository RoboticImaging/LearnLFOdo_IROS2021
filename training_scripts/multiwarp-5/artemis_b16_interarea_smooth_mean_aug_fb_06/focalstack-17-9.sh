python3 train_multiwarp.py focalstack \
/project/CIAutoInterp/data/module-1-1/module1-3-png \
artemis_b16_interarea_smooth_mean_aug_fb_06/multiwarp-5/focalstack-17-9 \
--save-path /project/CIAutoInterp/results \
--num-cameras 17 \
--num-planes 9 \
-c  3 8 13 7 9 \
--sequence-length 2 \
-b16 -s0.6 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-j16 \
--forward-backward \
--fb-loss-weight 0.3
