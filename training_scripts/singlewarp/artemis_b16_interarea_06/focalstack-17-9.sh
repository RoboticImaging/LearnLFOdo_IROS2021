python3 train_multiwarp.py focalstack \
/project/CIAutoInterp/data/module-1-1/module1-2-png \
artemis_test_b16_interarea_06/singlewarp/focalstack-17-9 \
--save-path /project/CIAutoInterp/results \
--num-cameras 17 \
--num-planes 9 \
-c 8 \
--sequence-length 2 \
-b16 -s0.6 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-j16 \
--pretrained-disp /project/CIAutoInterp/results/artemis_test_b16_interarea/singlewarp/focalstack-17-9/dispnet_20_checkpoint.pth.tar \
--pretrained-exppose /project/CIAutoInterp/results/artemis_test_b16_interarea/singlewarp/focalstack-17-9/posenet_20_checkpoint.pth.tar
