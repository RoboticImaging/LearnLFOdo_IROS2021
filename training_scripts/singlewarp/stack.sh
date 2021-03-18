python3 train_multiwarp.py stack \
~/Documents/thesis/epidata/module-1-1/module1-1-png \
final/singlewarp/monocular \
--save-path ~/Documents/thesis/checkpoints \
-c 8 \
--sequence-length 3 \
-b4 -s1.0 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
