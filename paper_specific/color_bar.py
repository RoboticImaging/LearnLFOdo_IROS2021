# This script plots a colorbar. Useful for putting results in the paper
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm
from matplotlib import colorbar
import cv2
import numpy as np

# fig = plt.figure(figsize=(1,8))
colormap = plt.get_cmap("viridis").reversed()

# ax = fig.add_axes([0.05, 0.05, 0.15, 0.15])
# img = cv2.imread("/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/code/paper_imgs/epi_without_stack_o_1_epoch_40.png")
# plt.imshow(img)
#
# ax = fig.add_axes([0.75, 0.05, 0.15, 0.9])
# norm = mpl.colors.Normalize(vmin=0.11, vmax=0.4)
# # c = mpl.colorbar.ColorbarBase(ax, cmap=colormap, norm=norm, orientation="vertical")
# ax.tick_params(axis="y", labelsize=14)


fig, ax = plt.subplots(1)
vmin = 0.25
vmax = 0.85
pcm = ax.pcolormesh(np.random.random((20, 20)) * (vmax - vmin) + vmin, cmap=colormap)
fig.colorbar(pcm, ax=ax)
plt.show()

plt.show()