python3 train_multiwarp.py focalstack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
final/singlewarp/focalstack-17-5 \
--save-path ~/Documents/thesis/checkpoints \
--num-cameras 17 \
--num-planes 5 \
-c 8 \
--sequence-length 3 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
