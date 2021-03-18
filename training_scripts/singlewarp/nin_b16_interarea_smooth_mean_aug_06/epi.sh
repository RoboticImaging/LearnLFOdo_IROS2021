python3 train_multiwarp.py epi \
/home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/module-1-1/module1-3-png \
nin_b16_interarea_smooth_mean_aug_06/singlewarp/epi/ \
-c 8 \
--save-path /home/dtejaswi/Documents/projects/student_projects/joe_daniel/results \
--sequence-length 2 \
-b8 -s0.6 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-e full \
-j12