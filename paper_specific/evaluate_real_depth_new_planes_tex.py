# This script generates violin plots of estimated depths for the experiment
# where a plane was placed at distances of 40, 50, 60, 70, 80 from the camera
# and compares it with ground truth data
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

import matplotlib.font_manager as fm

import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True

from pathlib import Path
fpath = Path(matplotlib.get_data_path(), "fonts/ttf/cmr10.ttf")
fprop = fm.FontProperties(fname=fpath)
matplotlib.rcParams['font.family'] = fprop.get_name()


def get_depth_array(path, r, c, h, w, visualize=False, title="depth"):
    """
    returns an array fo depth values within a region of interest,
    given the path to disparity data and the roi
    :param path: path to disparity data
    :type path: str
    :param r: row of top left corner of roi
    :type r: int
    :param c: column of top left corner of roi
    :type c: int
    :param h: height of the roi in pixels
    :type h: int
    :param w: width of the roi in pixels
    :type w: int
    :return: the array of depth values within the roi
    :rtype: array
    """
    img = np.load(path)
    img = 1.0 / img
    imgd = img[r:r+h, c:c+w]

    if visualize:
        colormap = plt.get_cmap("viridis").reversed()
        plt.figure()
        plt.suptitle(title)
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap=colormap)
        plt.clim(0.3, 0.8)
        # add rectangle to plot
        plt.gca().add_patch(Rectangle((c, r), w, h,
                               edgecolor='red',
                               facecolor='none',
                               fill=True,
                               lw=1))
        plt.subplot(1, 2, 2)
        plt.imshow(imgd, cmap=colormap)
        plt.clim(0.3, 0.8)
        # plt.show()
    return imgd


def get_roi(dist):
    """
    A hacky way to generate roi's given the name of the experiment (these values are hardcoded)
    :param dist: distance of the plane from the camera (20, 40, 60, 80, 100)
    :type dist: str
    :return: row, column of the top left corner of the roi and height and width of the roi in pixels
    :rtype: tuple of 4 values
    """
    # return 0, 0, 160, 224
    if dist == "400":
        return 80, 75, 65, 100
    # elif dist == "425":
    #     return 60, 60, 60, 80
    # elif dist == "450":
    #     return 60, 80, 50, 80
    # elif dist == "475":
    #     return 60, 80, 50, 70
    elif dist == "500":
        return 80, 60, 55, 75
    # elif dist == "525":
    #     return 60, 90, 50, 70
    # elif dist == "550":
    #     return 60, 90, 40, 60
    # elif dist == "575":
    #     return 60, 90, 40, 60
    elif dist == "600":
        return 80, 60, 45, 65
    # elif dist == "625":
    #     return 60, 95, 40, 55
    # elif dist == "650":
    #     return 65, 95, 40, 55
    # elif dist == "675":
    #     return 65, 100, 35, 50
    elif dist == "700":
        return 80, 75, 35, 45
    # elif dist == "725":
    #     return 65, 100, 30, 40
    # elif dist == "750":
    #     return 65, 100, 30, 40
    # elif dist == "775":
    #     return 65, 100, 30, 40
    elif dist == "800":
        return 80, 75, 30, 35
    else:
        raise ValueError("Incorrect dist entered")


def plot_violins_multiwarp(depth_vals_global, dists, outer_all=None):
    data_epi = []
    data_epi_nostack = []
    data_fs5 = []
    data_fs9 = []
    data_stack = []

    for dd in dists:
        d = str(dd)

        if outer_all == "outer":
            ####### Multiwarp-outer ##########
            data_epi.append(depth_vals_global[d + "_multiwarp-outer_epi"].flatten())
            data_epi_nostack.append(depth_vals_global[d + "_multiwarp-outer_epi_without_disp_stack"].flatten())
            data_fs5.append(depth_vals_global[d + "_multiwarp-outer_focalstack-17-5"].flatten())
            data_fs9.append(depth_vals_global[d + "_multiwarp-outer_focalstack-17-9"].flatten())
            data_stack.append(depth_vals_global[d + "_multiwarp-outer_stack"].flatten())
        elif outer_all == "all":
            ####### Multiwarp-all ##########
            data_epi.append(depth_vals_global[d + "_multiwarp-all_epi"].flatten())
            data_epi_nostack.append(depth_vals_global[d + "_multiwarp-all_epi_without_disp_stack"].flatten())
            data_fs5.append(depth_vals_global[d + "_multiwarp-all_focalstack-17-5"].flatten())
            data_fs9.append(depth_vals_global[d + "_multiwarp-all_focalstack-17-9"].flatten())
            data_stack.append(depth_vals_global[d + "_multiwarp-all_stack"].flatten())
        else:
            ####### Multiwarp-5 ##########
            data_epi.append(depth_vals_global[d + "_multiwarp-5_epi"].flatten())
            data_epi_nostack.append(depth_vals_global[d + "_multiwarp-5_epi_without_disp_stack"].flatten())
            data_fs5.append(depth_vals_global[d + "_multiwarp-5_focalstack-17-5"].flatten())
            data_fs9.append(depth_vals_global[d + "_multiwarp-5_focalstack-17-9"].flatten())
            data_stack.append(depth_vals_global[d + "_multiwarp-5_stack"].flatten())

    plt.figure()
    xvals = np.linspace(1, 5, len(dists)) * 0.6
    true_depth = np.array(dists).astype(np.float) / 1000.0

    violin_width = 0.1
    plt.violinplot(data_stack, positions=xvals - 0.2, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_fs5, positions=xvals - 0.1, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_fs9, positions=xvals, showextrema=True, showmeans=True, widths=violin_width)
    # plt.violinplot(data_epi, positions=xvals + 0.1, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_epi_nostack, positions=xvals + 0.2, showextrema=True, showmeans=True, widths=violin_width)
    plt.step(np.concatenate(([0], xvals))+0.3, np.concatenate(([0.4], true_depth)), 'k')

    # plt.xticks(xvals)
    # plt.xlim([0.3, 10.5])
    # plt.ylim([0.3, 0.85])

    ax = plt.gca()
    # Customize minor tick labels
    dists_str = ['0.4', '0.5', '0.6', '0.7', '0.8']

    ax.xaxis.set_major_locator(ticker.FixedLocator(xvals))
    # ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(dists_str))

    ax.xaxis.set_minor_locator(ticker.FixedLocator(xvals-0.3))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='x', which='major', length=0)
    plt.grid(axis='x', which='minor', linestyle='dashed')
    plt.title("Depth estimates in multi-warp reconstruction", fontsize=14)
    plt.xlabel("Distance of planar object from the EPIModule [m]", fontsize=14)
    plt.ylabel("Estimated distance [m]", fontsize=14)
    custom_patches = [Patch(facecolor='C0', edgecolor='None', label="volumetric stack"),
                      Patch(facecolor='C1', edgecolor='None', label="focalstack-5"),
                      Patch(facecolor='C2', edgecolor='None', label="focalstack-9"),
                      # Patch(facecolor='C3', edgecolor='None', label="epi"),
                      Patch(facecolor='C3', edgecolor='None', label="ours"),
                      Line2D([0], [0], color='k', lw=2, label='Ground truth depth')]
    ax.legend(handles=custom_patches, loc='lower right')

def plot_violins_singlewarp(depth_vals_global, dists):
    data_epi = []
    data_epi_nostack = []
    data_fs5 = []
    data_fs9 = []
    data_stack = []
    data_mono = []

    for dd in dists:
        d = str(dd)
        data_epi.append(depth_vals_global[d + "_singlewarp_epi"].flatten())
        data_epi_nostack.append(depth_vals_global[d + "_singlewarp_epi_without_disp_stack"].flatten())
        data_fs5.append(depth_vals_global[d + "_singlewarp_focalstack-17-5"].flatten())
        data_fs9.append(depth_vals_global[d + "_singlewarp_focalstack-17-9"].flatten())
        data_stack.append(depth_vals_global[d + "_singlewarp_stack"].flatten())
        data_mono.append(depth_vals_global[d + "_singlewarp_monocular"].flatten())

    plt.figure()
    xvals = np.linspace(1, 5, len(dists)) * 0.6
    true_depth = np.array(dists).astype(np.float) / 1000.0

    violin_width = 0.1
    plt.violinplot(data_stack, positions=xvals - 0.25, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_fs5, positions=xvals - 0.15, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_fs9, positions=xvals-0.05, showextrema=True, showmeans=True, widths=violin_width)
    # plt.violinplot(data_epi, positions=xvals + 0.05, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_epi_nostack, positions=xvals + 0.15, showextrema=True, showmeans=True, widths=violin_width)
    plt.violinplot(data_mono, positions=xvals + 0.25, showextrema=True, showmeans=True, widths=violin_width)
    plt.step(np.concatenate(([0], xvals)) + 0.3, np.concatenate(([0.4], true_depth)), 'k')

    # plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    # plt.xlim([0.3, 10.5])
    # plt.ylim([0,0.4])

    ax = plt.gca()
    # Customize minor tick labels
    dists_str = ['0.4', '0.5', '0.6', '0.7', '0.8']

    ax.xaxis.set_major_locator(ticker.FixedLocator(xvals))
    # ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(dists_str))

    ax.xaxis.set_minor_locator(ticker.FixedLocator(xvals - 0.3))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='x', which='major', length=0)
    plt.grid(axis='x', which='minor', linestyle='dashed')
    plt.title("Depth estimates in single-warp reconstruction", fontsize=14)
    plt.xlabel("Distance of planar object from the EPIModule [m]", fontsize=14)
    plt.ylabel("Estimated distance [m]", fontsize=14)

    custom_patches = [Patch(facecolor='C0', edgecolor='None', label="volumetric stack"),
                      Patch(facecolor='C1', edgecolor='None', label="focalstack-5"),
                      Patch(facecolor='C2', edgecolor='None', label="focalstack-9"),
                      # Patch(facecolor='C3', edgecolor='None', label="ours"),
                      Patch(facecolor='C3', edgecolor='None', label="ours"),
                      Patch(facecolor='C4', edgecolor='None', label="monocular"),
                      Line2D([0], [0], color='k', lw=2, label='Ground truth depth')]
    ax.legend(handles=custom_patches)


def main():

    distances = [400, 500, 600, 700, 800]

    ########### multiwarp-all #########
    # modes = ["multiwarp-all"]
    # root_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv_mean"
    # epoch = "40"

    ########## multiwarp-outer #########
    # modes = ["multiwarp-outer"]
    # root_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv_mean"
    # epoch = "60"

    # ########### multiwarp-5 #########
    modes = ["singlewarp", "multiwarp-5"]
    root_folder = "/home/dtejaswi/tensorboard_hpc/artemis_test_b16_interarea_tv_mean"
    epoch = "40"

    enc_multi = ["epi", "focalstack-17-5", "focalstack-17-9", "stack", "epi_without_disp_stack"]
    enc_single = ["epi", "focalstack-17-5", "focalstack-17-9", "stack", "monocular", "epi_without_disp_stack"]
    name = "planes_tex"

    # load all the depth values into a dictionary
    depth_vals_global = dict()
    mean_dvals_global  = dict()
    std_dvals_global = dict()
    for dd in distances:
        d = str(dd)
        r, c, h, w = get_roi(str(d))
        for mode in modes:
            if mode == "singlewarp":
                enc = enc_single
            else:
                enc = enc_multi
            for e in enc:
                if epoch == "":
                    # for backwards compatibility
                    disp_path = os.path.join(root_folder, mode, e,
                                             "results/" + name + d + "_1/disp/", "000000.npy")
                else:
                    # for use with a particular epoch

                    if mode == "multiwarp-outer" and e == "epi":
                        # special case for multiwarp outer and epi - epoch is 40
                        disp_path = os.path.join(
                            root_folder, mode, e,
                            "results/plane_" + name + "_" + d + "_1_epoch_" + "40" + "/disp/",
                            "000000.npy")
                    elif "interarea" in root_folder and mode == "multiwarp-5" and e == "epi_without_disp_stack":
                        # special case for multiwarp outer and epi - epoch is 40
                        disp_path = os.path.join(
                            root_folder, mode, e,
                            "results/plane_" + name + "_" + d + "_1_epoch_" + "60" + "/disp/",
                            "000000.npy")
                    else:
                        disp_path = os.path.join(
                            root_folder, mode, e,
                            "results/plane_" + name + "_" + d + "_1_epoch_" + epoch + "/disp/",
                            "000000.npy")

                dvals = get_depth_array(disp_path, r, c, h, w, visualize=False, title=mode+" "+e+" "+d)
                # dvals = get_depth_array(disp_path, r, c, h, w, visualize=False)
                key = d + "_" + mode + "_" + e
                depth_vals_global[key] = dvals

                mean_dval = np.mean(dvals)
                mean_dvals_global[key] = mean_dval

                std_dval = np.std(dvals)
                print("%s \n %.3f & %.3f"%(key, mean_dval, std_dval))
                std_dvals_global[key] = std_dval

    if "multiwarp-all" in modes:
        plot_violins_multiwarp(depth_vals_global, distances, outer_all="all")
    elif "multiwarp-outer" in modes:
        plot_violins_multiwarp(depth_vals_global, distances, outer_all="outer")
    else:
        plot_violins_multiwarp(depth_vals_global, distances)
        plot_violins_singlewarp(depth_vals_global, distances)

    plt.show()

if __name__ == "__main__":
    main()