python3 train_multiwarp.py stack \
/home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/module-1-1/module1-3-png \
nin_b16_interarea_smooth_mean_aug_06/multiwarp-5/stack \
--save-path /home/dtejaswi/Documents/projects/student_projects/joe_daniel/results \
-c 3 8 13 7 9 \
-k full \
--sequence-length 2 \
-b8 -s0.6 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-j12