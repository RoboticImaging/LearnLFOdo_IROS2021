# This script visualizes the depth predictions of the different methods
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import image as mpimg
import numpy as np
import os

def depth_image(path, border_crop = 10):
    """
    Reads a .npy file containing disparity output and converts it into a depth image output
    :param path: path to the .npy file
    :type path: str
    :return: depth image where the pixel values are in metres
    :rtype: numpy array
    """
    img = np.load(path)
    img = 1.0 / img
    h, w = img.shape
    img = img [border_crop:h-border_crop, border_crop:w-border_crop]
    return img

def plot_figure(image, img_index, fig_index, num_figs, colormap, title, cmin, cmax):
    plt.subplot(1, num_figs, fig_index)
    plt.imshow(image, cmap=colormap)
    plt.clim(cmin, cmax)

    mpimg.imsave("../paper_imgs/" + mode + "_" + title + "_" + seq + "_" + str(f) + ".png",
                 image, vmin=cmin, vmax=cmax, cmap=colormap)

    print(title, np.min(image), np.max(image))
    plt.title("{} ".format(img_index) + title)

fig = plt.figure()
colormap = plt.get_cmap("viridis").reversed()
ax = plt.gca()
# tickvals = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# RAL results - sequence length 3, bilinear downsampling, smoothness loss
# img_folder = "/home/dtejaswi/Desktop/joseph_daniel/ral/" + mode + "/"
# out_folder = "/home/dtejaswi/Desktop/joseph_daniel/ral/figs/" + mode + "/"

# Current framework - sequence length 2, bilinear downsampling, smoothness loss
# img_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16/" + mode + "/"
# out_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16/" + mode + "/"

# Current framework - sequence length 2, area average downsampling, smoothness loss
# img_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea/" + mode + "/"
# out_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea/" + mode + "/"

# Current framework - sequence length 2, area average downsampling, total variation aboslute loss
# img_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv/" + mode + "/"
# out_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv/" + mode + "/"

# Current framework - sequence length 2, area average downsampling, total variation square loss
# img_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_learn/" + mode + "/"
# out_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_learn/" + mode + "/"

# mode = "singlewarp"
mode = "multiwarp-5"
# mode = "multiwarp-outer"
img_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv_mean/" + mode + "/"
out_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv_mean/" + mode + "/"

# img_folder = "/media/dtejaswi/tejaswi1/joe_artemis_b16_copy/" + mode + "/"
# out_folder = "/media/dtejaswi/tejaswi1/joe_artemis_b16_copy/" + mode + "/"

max_img_index = 16
seq = "seq40_epoch_40"
f0 = "focalstack-17-5/results/"+seq
f1 = "focalstack-17-5/results/"+seq+"/disp"
f2 = "focalstack-17-9/results/"+seq+"/disp"
f3 = "epi/results/" + seq + "/disp"
f4 = "epi_without_disp_stack/results/" + seq + "/disp"
f5 = "stack/results/" + seq + "/disp"

if mode == "singlewarp":
    f6 = "monocular/results/" + seq + "/disp"

img_id = 14
# for f in range(0, 100):
for f in range(img_id, img_id+1):

    file = "{:06d}.npy".format(f)
    img1 = depth_image(os.path.join(img_folder, f1, file))
    img2 = depth_image(os.path.join(img_folder, f2, file))
    img3 = depth_image(os.path.join(img_folder, f3, file))
    img4 = depth_image(os.path.join(img_folder, f4, file))
    img5 = depth_image(os.path.join(img_folder, f5, file))
    if mode == "singlewarp":
        img6 = depth_image(os.path.join(img_folder, f6, file))

    if mode == "singlewarp":
        cmin = 0.11
        cmax = 0.4
    else:
        cmin = 0.27
        cmax = 0.85

    if mode == "singlewarp":
        num_figs = 7
    else:
        num_figs = 6
    plot_figure(img1, f, 1, num_figs, colormap, "fs-17-5", cmin, cmax)
    plot_figure(img2, f, 2, num_figs, colormap, "fs-17-9", cmin, cmax)
    plot_figure(img3, f, 3, num_figs, colormap, "epi", cmin, cmax)
    plot_figure(img4, f, 4, num_figs, colormap, "epi_without_stack", cmin, cmax)
    plot_figure(img5, f, 5, num_figs, colormap, "stack", cmin, cmax)
    if mode == "singlewarp":
        plot_figure(img6, f, 6, num_figs, colormap, "monocular", cmin, cmax)

    color_file = "{:06d}.png".format(f)
    img = cv2.imread(os.path.join(img_folder, f0, color_file))
    height = img.shape[0]
    width = img.shape[1]
    bound = 2
    img = img[bound:height-bound, bound:width-bound]

    plt.subplot(1,7,7)
    plt.imshow(img)
    # print("f: {} min_depth: {} max_depth: {}".format(f, np.min(img), np.max(img)))

    plt.draw()
    # plt.pause(0.5)
    plt.show()