python3 train_multiwarp.py stack \
/project/CIAutoInterp/data/module-1-1/module1-2-png \
artemis_test_b16_interarea_tv/multiwarp-outer/stack \
--save-path /project/CIAutoInterp/results \
-c 1 8 16 5 12 \
-k full \
--sequence-length 2 \
-b16 -s0.3 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 100 \
--log-output \
--gray \
-j16 \
--pretrained-disp /project/CIAutoInterp/results/artemis_test_b16_interarea/multiwarp-5/stack/dispnet_20_checkpoint.pth.tar \
--pretrained-exppose /project/CIAutoInterp/results/artemis_test_b16_interarea/multiwarp-5/stack/posenet_20_checkpoint.pth.tar
