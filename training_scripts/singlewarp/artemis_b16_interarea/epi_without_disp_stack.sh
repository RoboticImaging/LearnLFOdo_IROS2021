python3 train_multiwarp.py epi \
/project/CIAutoInterp/data/module-1-1/module1-2-png \
artemis_test_b16_interarea/singlewarp/epi_without_disp_stack/ \
-c 8 \
--save-path /project/CIAutoInterp/results \
--sequence-length 2 \
-b16 -s0.3 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-e full \
-j16 \
--without-disp-stack