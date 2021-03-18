import cv2
import numpy as np
import matplotlib.pyplot as plt

orig_image = cv2.imread("/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png/seq3/8/0000000000.png")
planes_image = cv2.imread("/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png/planes_tex/400/seq1/8/0000000000.png")

orig_black = np.zeros((orig_image.shape[0], orig_image.shape[1]))
planes_black = np.zeros((planes_image.shape[0], planes_image.shape[1]))
overlayed_image = np.zeros((planes_image.shape[0], planes_image.shape[1], 3), dtype=np.int)

for r in range(0, orig_black.shape[0]):
    for c in range(0, orig_image.shape[1]):
        if orig_image[r,c,0]==0 and orig_image[r,c,1]==0 and orig_image[r,c,2]==0:
            orig_black[r, c] = 255
            overlayed_image[r, c, 0] = 255

        if planes_image[r,c,0]==0 and planes_image[r,c,1]==0 and planes_image[r,c,2]==0:
            planes_black[r, c] = 255
            overlayed_image[r, c, 1] = 255

fig = plt.figure()
plt.imshow(orig_black)
# plt.set_cmap("gray")

fig = plt.figure()
plt.imshow(planes_black)
# plt.set_cmap("gray")

fig = plt.figure()
plt.imshow(overlayed_image)

plt.show()