python3 train_multiwarp.py epi \
/home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/module-1-1/module1-1-png \
final/singlewarp/epi/ \
-c 8 \
--save-path /home/dtejaswi/Documents/projects/student_projects/joe_daniel/data/checkpoints \
--sequence-length 3 \
-b4 -s0.3 -m0.0 -g0.0 \
--epochs 300 \
--log-output \
--gray