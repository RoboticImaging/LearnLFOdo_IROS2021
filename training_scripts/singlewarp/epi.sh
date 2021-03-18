python3 train_multiwarp.py epi \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
final/singlewarp/epi/ \
-c 8 \
--save-path ~/Documents/thesis/checkpoints \
--sequence-length 3 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
