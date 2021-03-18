import epi_io
import os
import cv2
import shutil

# ******************************************************
# Script that converts epirect images to png images.
# This code used CV.INTER_AREA interpolation
# USAGE:
# ******************************************************
# The epirect images should be located in folders named seq#, where # is the number of the sequence.
# Set the path to the input and output folders
# Set the sequences that need to be converted

input_root = "/media/dtejaswi/Seagate Expansion Drive/JoeDanielThesisData/data/sequences"
output_root = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1"
# sequences = [50, 51, 52]
sequences = range(2, 46)
is_seq_prefix = True

for sequence in sequences:
    if is_seq_prefix:
        input_drectory = os.path.join(input_root, "seq" + str(sequence))
        output_directory = os.path.join(output_root, "module1-2-png", "seq" + str(sequence))
    else:
        input_drectory = os.path.join(input_root, str(sequence))
        output_directory = os.path.join(output_root, "module1-2-png", str(sequence))

    print("epirect directory: {}".format(input_drectory))
    print("output directory: {}".format(output_directory))

    # get a list of all the epirect images
    lfs = os.listdir(input_drectory)
    lfs = [int(lf.split(".")[-2]) for lf in lfs if lf.split(".")[-1] == "epirect"]
    lfs = sorted(lfs)
    lfs = ["{:06d}.epirect".format(lf) for lf in lfs]
    lfs = [os.path.join(input_drectory, lf) for lf in lfs]

    # go through the list of epirect images and convert them to png
    for i, lf in enumerate(lfs):
        print(f"{i}/{len(lfs)-1}", end="\r")
        lf_img = epi_io.read_rectified(lf)
        for j in range(0, 17):
            img = lf_img[j, :, :, :]
            output_directory_numbered = os.path.join(output_directory, str(j))

            if not os.path.exists(output_directory_numbered):
                os.makedirs(output_directory_numbered)

            output_filename = os.path.join(output_directory_numbered, "{:010d}.png".format(i))
            img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_filename, img)

    if os.path.exists(os.path.join(input_drectory, "poses_gt_absolute.npy")):
        print("Copying GT absolute poses")
        shutil.copyfile(os.path.join(input_drectory, "poses_gt_absolute.npy"),
                        os.path.join(output_directory, "poses_gt_absolute.npy"))
    else:
        print("GT absolute poses do not exist")

    if os.path.exists(os.path.join(input_drectory, "poses_gt_relative.npy")):
        print("Copying GT relative poses")
        shutil.copyfile(os.path.join(input_drectory, "poses_gt_relative.npy"),
                        os.path.join(output_directory, "poses_gt_relative.npy"))
    else:
        print("GT relative poses do not exist")
