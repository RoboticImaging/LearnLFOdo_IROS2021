# This script plots a histogram of the training trajectories just for debugging
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_5x9_grid(data, title):
    fig, axs = plt.subplots(5, 9)
    plt.suptitle(title)
    for fid in range(0, 44):
        print(fid)
        axs[fid // 9, fid % 9].hist(data[fid])
        axs[fid // 9, fid % 9].set_title("seq{}".format(fid))
        # axs[fid // 9, fid % 9].get_xaxis().set_visible(False)
        # axs[fid // 9, fid % 9].get_yaxis().set_visible(False)
    # plt.subplots_adjust(top=0.90, bottom=0.1, right=0.9, left=0.1,
    #                     hspace=0.6, wspace=0.1)
    # plt.margins(0, 0)

def get_data(sequences, data_dir):
    x = []
    y = []
    z = []
    roll = []
    pitch = []
    yaw = []
    for seq in sequences:
        input_file = os.path.join(data_dir, "seq" + str(seq), "poses_gt_relative.npy")

        poses = np.load(input_file)
        x.append(poses[:, 0])
        y.append(poses[:, 1])
        z.append(poses[:, 2])
        roll.append(poses[:, 3])
        pitch.append(poses[:, 4])
        yaw.append(poses[:, 5])

    # plot_5x9_grid(x, "x")
    # plot_5x9_grid(y, "y")
    # plot_5x9_grid(z, "z")
    # plot_5x9_grid(roll, "roll")
    # plot_5x9_grid(pitch, "pitch")
    # plot_5x9_grid(yaw, "yaw")

    all_x = np.concatenate(x).ravel()
    all_y = np.concatenate(y).ravel()
    all_z = np.concatenate(z).ravel()
    all_roll = np.concatenate(roll).ravel()
    all_pitch = np.concatenate(pitch).ravel()
    all_yaw = np.concatenate(yaw).ravel()

    return all_x, all_y, all_z, all_roll, all_pitch, all_yaw

def get_min_max(a, b):
    minab = np.min([np.min(a), np.min(b)])
    maxab = np.max([np.max(a), np.max(b)])
    return minab, maxab

def plot_hist_data(a, b, title):
    plt.figure()
    minab, maxab = get_min_max(a, b)
    bins = np.linspace(minab, maxab, 100)
    plt.title(title)
    plt.hist(a, bins=bins, label="train")
    plt.hist(b, bins=bins, label="test")
    plt.ylim([0,200])

if __name__ == "__main__":
    data_dir = "/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/data/module-1-1/module1-2-png/"

    validation_sequences = [16, 40, 44, 28]
    valid_x, valid_y, valid_z, valid_roll, valid_pitch, valid_yaw = get_data(validation_sequences, data_dir)

    sequences = [*range(2, 46)]
    sequences.pop(14)
    sequences.pop(25)
    sequences.pop(36)
    sequences.pop(39)
    train_x, train_y, train_z, train_roll, train_pitch, train_yaw = get_data(sequences, data_dir)


    sequences = [60, 61, 62]
    test_x, test_y, test_z, test_roll, test_pitch, test_yaw = get_data(sequences, data_dir)

    # plot_hist_data(train_x, test_x, "x")
    # plot_hist_data(train_y, test_y, "y")
    # plot_hist_data(train_z, test_z, "z")
    # plot_hist_data(train_roll, test_roll, "roll")
    # plot_hist_data(train_pitch, test_pitch, "pitch")
    # plot_hist_data(train_yaw, test_yaw, "yaw")
    
    plot_hist_data(train_x, valid_x, "x")
    plot_hist_data(train_y, valid_y, "y")
    plot_hist_data(train_z, valid_z, "z")
    plot_hist_data(train_roll, valid_roll, "roll")
    plot_hist_data(train_pitch, valid_pitch, "pitch")
    plot_hist_data(train_yaw, valid_yaw, "yaw")
    
    plt.show()
