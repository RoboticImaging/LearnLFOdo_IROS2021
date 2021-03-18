python3 train_multiwarp.py focalstack \
/home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/module-1-1/module1-3-png \
nin_b16_interarea_smooth_mean_aug_06/singlewarp/focalstack-17-5 \
--save-path /home/dtejaswi/Documents/projects/student_projects/joe_daniel/results \
--num-cameras 17 \
--num-planes 5 \
-c 8 \
--sequence-length 2 \
-b8 -s0.6 -m0.0 -g0.0 \
--lr 5e-4 \
--epochs 200 \
--log-output \
--gray \
-j12