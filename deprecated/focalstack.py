import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread


def load_as_float(path, gray):
    print("WARNING focalstack.load_as_float IS DEPRECATED")
    im = imread(path).astype(np.float32)
    if gray:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return im


def load_lightfield(path, cameras, gray):
    print("WARNING focalstack.load_lightfield IS DEPRECATED")
    imgs = []
    for cam in cameras:
        img_path = path.replace('/8/', '/{}/'.format(cam))
        imgs.append(load_as_float(img_path, gray))
    return imgs


def shift_sum(lf, shift, dof, gray):
    print("WARNING focalstack.shift_sum IS DEPRECATED")
    if type(lf) is list:
        lf = np.array(lf)
    assert (lf.shape[0] == 17)

    if dof == 17:
        left = [4, 5, 6, 7]
        right = [9, 10, 11, 12]
        top = [0, 1, 2, 3]
        bottom = [13, 14, 15, 16]
    elif dof == 13:
        left = [5, 6, 7]
        right = [9, 10, 11]
        top = [1, 2, 3]
        bottom = [13, 14, 15]
    elif dof == 9:
        left = [6, 7]
        right = [9, 10]
        top = [2, 3]
        bottom = [13, 14]
    elif dof == 5:
        left = [7]
        right = [9]
        top = [3]
        bottom = [13]
    else:
        raise ValueError("Cannot focus at depth {}".format(dof))

    focalstack = lf[8].astype(np.float32)
    if shift == 0:
        focalstack += np.sum(lf[left + right + top + bottom], 0)
    else:
        for i in left:
            img = lf[i]
            s = (8 - i) * shift
            focalstack[:, :-s:, :] += img[:, s:, :]
        for i in right:
            img = lf[i]
            s = (i - 8) * shift
            focalstack[:, s:, :] += img[:, :-s, :]
        for i in top:
            img = lf[i]
            s = (4 - i) * shift
            focalstack[:-s, :, :] += img[s:, :, :]
        for i in bottom:
            img = lf[i]
            s = (i - 12) * shift
            focalstack[s:, :, :] += img[:-s, :, :]

    focalstack = focalstack / dof
    if gray:
        focalstack = cv2.cvtColor(focalstack, cv2.COLOR_RGB2GRAY)
    return focalstack.astype(np.uint8)


def load_multiplane_focalstack(path, numPlanes, numCameras, gray):
    print("WARNING focalstack.load_multiplane_focalstack is DEPRECATED")
    assert numCameras in [5, 9, 13, 17]
    assert numPlanes in [9, 7, 5, 3]

    planes = None
    if numPlanes == 9:
        planes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif numPlanes == 7:
        planes = [0, 1, 2, 3, 4, 6, 8]
    elif numPlanes == 5:
        planes = [0, 2, 4, 6, 8]
    elif numPlanes == 3:
        planes = [0, 4, 8]

    stacks = []
    lf = load_lightfield(path, gray=False, cameras=list(range(0, 17)))
    for p in planes:
        stacks.append(shift_sum(lf, p, numCameras, gray=gray))

    return stacks


