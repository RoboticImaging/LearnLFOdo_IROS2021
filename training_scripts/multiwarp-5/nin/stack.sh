python3 train_multiwarp.py stack \
/home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/module-1-1/module1-1-png \
final/multiwarp-5/stack \
--save-path /home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/checkpoints \
-c 3 8 13 7 9 \
--sequence-length 3 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray
