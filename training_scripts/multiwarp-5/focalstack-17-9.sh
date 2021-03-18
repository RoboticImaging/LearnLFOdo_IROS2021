python3 train_multiwarp.py focalstack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
final/multiwarp-5/focalstack-17-9 \
--save-path ~/Documents/thesis/checkpoints \
--num-cameras 17 \
--num-planes 9 \
-c  3 8 13 7 9 \
--sequence-length 3 \
-b4 -s0.1 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
