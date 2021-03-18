# This script visualizes the disparity images as predicted by the different methods
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

img_folder = "/home/dtejaswi/Desktop/joseph_daniel/ral/multiwarp-5/focalstack-17-5/results/51/disp"

plt.figure()

for f in range(0, 69):
    file = "{:06d}.npy".format(f)
    # img = cv2.imread(os.path.join(img_folder, file), cv2.CV_32FC1)
    img = np.load(os.path.join(img_folder, file))
    img = 1.0/ img
    height, width = img.shape
    bound = 2

    img = img[bound:height-bound, bound:width-bound]
    print("f: {} min_depth: {} max_depth: {}".format(f, np.min(img), np.max(img)))
    plt.imshow(img, cmap=plt.get_cmap("Wistia"))
    if f == 1:
        plt.colorbar()

    plt.draw()
    plt.pause(0.01)


