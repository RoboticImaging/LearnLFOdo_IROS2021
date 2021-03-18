python3 train_multiwarp.py focalstack \
/home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/module-1-1/module1-1-png \
final/multiwarp-5/focalstack-17-9 \
--save-path /home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/checkpoints \
--num-cameras 17 \
--num-planes 9 \
-c  3 8 13 7 9 \
--sequence-length 3 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
