import cv2
import numpy as np
import glob
import os

input_root = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png/"
sequences = range(2,46)
# sequences = range(10,46)
# sequences = [50, 51, 52]

for seq in sequences:
    central_folder = os.path.join(input_root, "seq"+str(seq), "8")  # 8 is the central camera
    output_folder = os.path.join(input_root, "seq"+str(seq))

    print("sequence: {}".format(seq))

    img_array = []
    input_files = os.listdir(central_folder)
    input_files = sorted(input_files)
    for i, filename in enumerate(input_files):
        print("{}/{}".format(i, len(input_files)-1), end="\r")
        if filename.split(".")[-1] == "png":
            img = cv2.imread(os.path.join(central_folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    out_filename = os.path.join(output_folder, "seq_" + str(seq) + ".avi")
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()