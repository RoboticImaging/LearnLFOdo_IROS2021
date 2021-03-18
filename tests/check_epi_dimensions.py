# This script checks the shapes of the EPI images and their stacking
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import numpy as np


def hori():
    num_images = 4
    lf_h = []
    for i in range(num_images):
        lf_h.append(np.random.randint(0, 9, [2, 3]))
    lf_h = np.array(lf_h)
    # print to see the dimensions. this should be num_cameras x num_rows x num_columns
    print("lf_h shape: {}".format(lf_h.shape))
    print(lf_h)

    # re-order the array to be of shape num_rows x num_cameras x num_columns
    lf_h2 = lf_h.transpose(1, 0, 2)
    print("lf_v2 shape: {}".format(lf_h2.shape))
    print(lf_h2)

    # reshape the array to be of shape 1 x (num_rows * num_cameras) x num_columns
    lf_h3 = lf_h2.reshape(1, lf_h2.shape[0] * lf_h2.shape[1], lf_h2.shape[2])
    print("lf_v3 shape: {}".format(lf_h3.shape))
    print(lf_h3)

def vert():
    num_images = 4

    # create a sample vertical lf
    lf_v = []
    for i in range(num_images):
        lf_v.append(np.random.randint(0, 9, [2, 3]))
    lf_v = np.array(lf_v)
    # print to see the dimensions. this should be num_cameras x num_rows x num_columns
    print("lf_v shape: {}".format(lf_v.shape))
    print(lf_v)

    # re-order the array to be of shape num_columns x num_cameras x num_rows
    lf_v2 = lf_v.transpose(2, 0, 1)
    print("lf_v2 shape: {}".format(lf_v2.shape))
    print(lf_v2)

    # reshape the array to be of shape (num_columns * num_cameras) x num_rows x 1
    lf_v3 = lf_v2.reshape(lf_v2.shape[0] * lf_v2.shape[1], lf_v2.shape[2], 1)
    print("lf_v3 shape: {}".format(lf_v3.shape))
    print(lf_v3)

    # finally take another transpose to reshape the array to 1 x num_rows x (num_columns * num_cameras)
    lf_v4 = lf_v3.transpose()
    print("lf_v4 shape: {}".format(lf_v4.shape))
    print(lf_v4)

if __name__ == "__main__":
    hori()
