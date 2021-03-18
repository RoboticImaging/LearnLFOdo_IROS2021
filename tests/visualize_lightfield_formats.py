# This script visualizes the different light field formats
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import matplotlib.pyplot as plt
import epimodule


def main():
    image_path = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-3-png/seq103/8/0000000040.png"
    save_figs = False

    # load and display focal stacks
    num_planes = 9
    num_cameras = 17
    lf_focalstack = epimodule.load_multiplane_focalstack(image_path,
                                                         num_planes=num_planes,
                                                         num_cameras=num_cameras,
                                                         gray=False)
    # --> Display all the shifted and averaged images
    fig = plt.figure()
    plt.suptitle("Focal Stack of {} planes of focus and {} cameras".format(num_planes, num_cameras))
    for i in range(num_planes):
        plt.subplot(3, 3, i+1)
        plt.title("Shift = {}".format(i))
        plt.imshow(lf_focalstack[i])
        plt.gca().set_axis_off()
        if save_figs:
            plt.imsave("fs_{}.png".format(i), lf_focalstack[i])

    lf_hori = epimodule.load_lightfield_horizontal_cameras(image_path)
    fig = plt.figure()
    plt.suptitle("Horizontal camera images")
    for i in range(8):      # 8 horizontal cameras
        plt.subplot(1, 8, i + 1)
        plt.imshow(lf_hori[i], cmap="gray")
        plt.gca().set_axis_off()
    lf_vert = epimodule.load_lightfield_vertical_cameras(image_path)
    fig = plt.figure()
    plt.suptitle("Vertical camera images")
    for i in range(8):  # 8 vertical cameras
        plt.subplot(8, 1, i + 1)
        plt.imshow(lf_vert[i], cmap="gray")

    # load and display tiled epipolar plane images
    lf_tiled = epimodule.load_tiled_epi_vertical(image_path)
    fig = plt.figure()
    # plt.imsave("lfv.png", lf_tiled[0,:,:])
    plt.suptitle("Vertically tiled epipolar images")
    plt.imshow(lf_tiled[0,:,:], cmap="gray")
    plt.gca().set_axis_off()
    if save_figs:
        plt.savefig("lfv.png")

    lf_tiled_h = epimodule.load_tiled_epi_horizontal(image_path)
    fig = plt.figure()
    plt.suptitle("horizontally tiled epipolar images")
    # plt.imsave("lfh.png", lf_tiled_h[0, :, :])
    plt.imshow(lf_tiled_h[0, :, :], cmap="gray")
    plt.gca().set_axis_off()
    if save_figs:
        plt.savefig("lfh.png")

    # tf_tiled_v, tf_tiles_h = epimodule.load_tiled_epi_full(image_path)

    # load and display stacked epipolar plane images
    lf_stacked = epimodule.load_stacked_epi(image_path, patch_size=160, same_parallax=False)
    print("Size of stacked image is {}x{}x{}".format(lf_stacked.shape[0], lf_stacked.shape[1], lf_stacked.shape[2]))
    fig = plt.figure()
    plt.suptitle("Stacking of camera images")
    for i in range(16):  # 16 cameras - BEWARE camera 8 (central) is appears twice
        plt.subplot(4, 4, i + 1)
        plt.imshow(lf_stacked[i], cmap="gray")
        plt.title(i)
        # plt.imsave("img_{}.png".format(i), lf_stacked[i])

    plt.show()


if __name__ == "__main__" :
    main()
