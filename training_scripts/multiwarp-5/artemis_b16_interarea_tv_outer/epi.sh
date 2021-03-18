python3 train_multiwarp.py epi \
/project/CIAutoInterp/data/module-1-1/module1-2-png \
artemis_test_b16_interarea_tv/multiwarp-outer/epi/ \
-c 1 8 16 5 12 \
--save-path /project/CIAutoInterp/results \
--sequence-length 2 \
-b16 -s0.3 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 100 \
--log-output \
--gray \
-e full \
-j16 \
--pretrained-disp /project/CIAutoInterp/results/artemis_test_b16_interarea/multiwarp-5/epi/dispnet_20_checkpoint.pth.tar \
--pretrained-exppose /project/CIAutoInterp/results/artemis_test_b16_interarea/multiwarp-5/epi/posenet_20_checkpoint.pth.tar