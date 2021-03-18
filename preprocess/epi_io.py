import numpy as np
import requests
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RECTIFIED_IMAGE_SHAPE = [17, 960, 1280, 3]

def write_rectified(LF, outfile):
    if type(LF) is np.ndarray: LF = LF.tobytes() 
    elif type(LF) is bytes: pass 
    else: raise TypeError("Expected ndarray or bytes, not {}".format(type(LF))) 
    
    with open(outfile, 'wb') as output:
        output.write(LF)
        print("LF saved as {}".format(outfile))


def read_rectified(path):
    return _from_bytes(path, shape=RECTIFIED_IMAGE_SHAPE)


def get_intrinsics():
    intrinsics = os.path.join(os.path.dirname(__file__), "intrinsics.txt")
    k = np.loadtxt(intrinsics)
    assert(k.shape == (3,3))
    return k


def capture_rectified(ip, outfile=None):
    pattern = "http://{}/frame/rectified".format(ip) 
    print("[CaptureRectified] Sending request...")
    data = requests.get(pattern, timeout=20)
    print("[CaptureRectified] Captured {} bytes".format(len(data.content)))

    if outfile is not None:
        write_rectified(data.content, outfile)
    return data.content


def save_image_array(LF, name="lf", encoding="png"):
    """ Epimodule image to 17 png's """
    for i in range(0, LF.lf.shape[0]):
        image = LF.lf[i, :, :, :]
        outfile = "{}_{}.{}".format(name, i, encoding)
        cv2.imwrite(outfile, image)


def keyboard_pan(lf):
    def handle_keystroke(k, i):
        keys = {"w": 119, "s": 115, "a": 97, "d": 100, "esc": 27}
        updown = [16, 15, 14, 13, 8, 3, 2, 1, 0]
        leftright = [12, 11, 10, 9, 8, 7, 6, 5, 4]
        if i in updown and k == keys["s"]:
            if(i < lf.shape[0]-1): i = updown[updown.index(i)-1]
        elif i in updown and k == keys["w"]:
            if(i > 0): i = updown[updown.index(i)+1]
        elif i in leftright and k == keys["a"]:
            if(i > 4): i = leftright[leftright.index(i)+1]
        elif i in leftright and k == keys["d"]:
            if(i < 12): i = leftright[leftright.index(i)-1]
        elif i in leftright:
            if k == keys["w"]: i = 3
            if k == keys["s"]: i = 13
        elif i in updown:
            if k == keys["d"]: i = 9
            if k == keys["a"]: i = 7
        return i

    print("Use A, S, W, D keys to pan light field.")
    print("Press esc to exit.")
    current_image = 8
    cv2.namedWindow('EPImage', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EPImage', 300, 300)
    while(True):
        cv2.imshow("EPImage", lf[current_image])
        key = cv2.waitKey(0)
        if(key == 27):
            break
        else:
            current_image = handle_keystroke(key, current_image)
    cv2.destroyAllWindows()


def _from_bytes(lf_bytes, shape, packing=np.uint8):
    if type(lf_bytes) is str:
        lf_bytes = open(lf_bytes, 'rb').read()
    lf = np.frombuffer(lf_bytes, packing)
    lf = np.reshape(lf, shape)
    return lf


def epi_histeq(lf, upper_percentile, lower_percentile):
    lf_size = lf.shape
    n_channels = lf_size[-1]

    lf = np.reshape(lf, [np.prod(lf_size[:-1]), 1, n_channels])
    lf = lf-np.min(lf)
    lf = (lf/np.max(lf)*255).astype(np.uint8)

    lf = cv2.cvtColor(lf, cv2.COLOR_RGB2HSV)
    lf_hs = lf[:, :, 0:2]
    lf_v = lf[:, :, 2]

    upper = np.percentile(lf_v, upper_percentile)
    lower = np.percentile(lf_v, lower_percentile)

    lf_v = np.maximum(lower, np.minimum(upper, lf_v))
    lf_v = (((lf_v-lower)/(upper - lower))*255).astype(np.uint8)

    if lf_hs.any():
        lf_v = np.reshape(lf_v, [np.prod(lf_size[:-1]), 1, 1])
        lf = np.concatenate([lf_hs, lf_v], 2)
        lf = cv2.cvtColor(lf, cv2.COLOR_HSV2RGB)

    return np.reshape(lf, lf_size)


def plotTrajectory(rotations, translations):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plt.show()

def plotCamera(ax, rotation, translation):

    u = rotation   
    
    ax.quiver(0, 0, 0, 0.1, 0, 0, length=1, normalize=True)
    plt.show()