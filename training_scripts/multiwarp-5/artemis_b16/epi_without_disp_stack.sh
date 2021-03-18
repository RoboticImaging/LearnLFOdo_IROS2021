python3 train_multiwarp.py epi \
/project/CIAutoInterp/data/module-1-1/module1-1-png \
artemis_test_b16/multiwarp-5/epi_without_disp_stack/ \
-c 3 8 13 7 9 \
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