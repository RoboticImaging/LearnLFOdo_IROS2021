# This script visualizes the weights, computed features and training in general
#
# Author and Maintainer: Tejaswi Digumarti (tejaswi.digumarti@sydney.edu.au)

import os
import torch
import custom_transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiwarp_dataloader import get_validation_focal_stack_loader, get_validation_stacked_lf_loader, get_validation_epi_loader
from lfmodels import LFDispNetViz as DispNetS
from lfmodels import LFPoseNetViz as PoseNet
from lfmodels import EpiEncoder, RelativeEpiEncoder
from loss_functions import multiwarp_photometric_loss
from utils import load_config


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, type=str, help="Folder with Pkl file containing training configuration")
    parser.add_argument("--seq", required=True, type=str, help="Name of sequence to perform inference on")
    parser.add_argument("--suffix", required=False, type=str, help="Additional suffix", default="")
    parser.add_argument("--data_dir", required=True, type=str, help="Folder where png files are present",
                        default="/home/dtejaswi/Desktop/joseph_daniel/extras/png/A/60")
    parser.add_argument("--sequence_length", required=True, type=int, help="sequence length for training - including target and reference images (2 or 3)")
    parser.add_argument("--no_pose",required=False, action="store_true", help="if set then input pose is zero. Use when inferring only depth")
    parser.add_argument("--use_checkpoint_at", required=False, type=str, help="Use the checkpoint at a particular epoch")
    parser.add_argument("--use_latest_not_best", required=False, action="store_true",
                        help="Use the latest set of weights rather than the best")
    return parser

def plot_4x4_grid(data, title):
    fig, axs = plt.subplots(4, 4)
    plt.suptitle(title)
    for fid in range(0, 16):
        axs[fid // 4, fid % 4].imshow(data[fid, 0, :, :])
        axs[fid // 4, fid % 4].set_title("filter: {}".format(fid))
        axs[fid // 4, fid % 4].get_xaxis().set_visible(False)
        axs[fid // 4, fid % 4].get_yaxis().set_visible(False)
    plt.subplots_adjust(top=0.90, bottom=0.1, right=0.9, left=0.1,
                        hspace=0.6, wspace=0.1)
    plt.margins(0, 0)

def plot_5x8_grid(axs, data):
    for ch in range(0, data.shape[0]):
        plt.imsave("/media/dtejaswi/DATA/CommonData/Projects/student_projects/joe_daniel/raw_for_paper/encoded/{}.png".format(ch), data[ch, :, :])

        axs[ch // 8, ch % 8].imshow(data[ch, :, :])
        axs[ch // 8, ch % 8].set_title("output: {}".format(ch))
        axs[ch // 8, ch % 8].get_xaxis().set_visible(False)
        axs[ch // 8, ch % 8].get_yaxis().set_visible(False)

def visualize_dispnet_encoder_weights(weights):
    # disparity encoder for epi images - vertical EPI
    disp_encoder_vert_weights = weights['encoder.conv1.weight'].detach().cpu().numpy()
    plot_4x4_grid(disp_encoder_vert_weights, "8x8 conv kernels applied on vertical epi images for disparity")
    # disparity encoder for epi images - horizontal EPI
    disp_encoder_hori_weights = weights['encoder.conv1_ex.weight'].detach().cpu().numpy()
    plot_4x4_grid(disp_encoder_hori_weights, "8x8 conv kernels applied on horizontal epi images for disparity")


def visualize_posenet_encoder_weights(weights):
    # posenet encoder for epi images - vertical EPI
    pose_encoder_vert_weights = weights['encoder.conv1.weight'].detach().cpu().numpy()
    plot_4x4_grid(pose_encoder_vert_weights, "8x8 conv kernels applied on vertical epi images for pose")
    # posenet encoder for epi images - horizontal EPI
    pose_encoder_hori_weights = weights['encoder.conv1_ex.weight'].detach().cpu().numpy()
    plot_4x4_grid(pose_encoder_hori_weights, "8x8 conv kernels applied on horizontal epi images for pose")


def visualize_dispnet_conv_weights(weights):
    np_weights = weights['conv1.0.weight'].detach().cpu().numpy()
    monkey=2

@torch.no_grad()
def main():
    args = get_parser().parse_args()
    config = load_config(os.path.join(args.config_dir, "config.pkl"))
    is_monocular = False

    # add missing param to multiwarp stack run
    # if "multiwarp-5" in config.name:
    #     if "/stack" in config.name:
    #         config.cameras_stacked="input"

    # multiwarp-5
    if "multiwarp-5" in config.name:
        mode = "multiwarp-5"
    elif "singlewarp" in config.name:
        mode = "singlewarp"
    else:
        raise ValueError("unknown mode")

    if "/epi" in config.name:
        enc = "epi"
    elif "/stack" in config.name:
        enc = "stack"
    elif "/focalstack-17-5" in config.name:
        enc = "focalstack-17-5"
    elif "/focalstack-17-9" in config.name:
        enc = "focalstack-17-9"
    elif "/monocular" in config.name:
        enc = "monocular"
        is_monocular = True
    else:
        raise ValueError("unknown encoding")

    if args.no_pose:
        print("Considering zero ground truth pose. Only depth is predicted correctly in this case.")

    ## data on which the pipeline has to be run
    # config.data = "/home/dtejaswi/Documents/Projects/student_projects/joseph_daniel/data/module-1-1/module1-1-png"
    # config.data = "/home/dtejaswi/Desktop/joseph_daniel/extras/png/A/60"
    config.data = args.data_dir

    if args.use_checkpoint_at is not None:
        config.posenet = os.path.join(args.config_dir, "posenet_" + args.use_checkpoint_at + "_checkpoint.pth.tar")
        config.dispnet = os.path.join(args.config_dir, "dispnet_" + args.use_checkpoint_at + "_checkpoint.pth.tar")
    else:
        # load configuration from checkpoints
        if args.use_latest_not_best:
            config.posenet = os.path.join(args.config_dir, "posenet_checkpoint.pth.tar")
            config.dispnet = os.path.join(args.config_dir, "dispnet_checkpoint.pth.tar")
        else:
            config.posenet = os.path.join(args.config_dir, "posenet_best.pth.tar")
            config.dispnet = os.path.join(args.config_dir, "dispnet_best.pth.tar")

    # define transformations
    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    disp_encoder = None
    pose_encoder = None

    # Load validation dataset loaders
    # --> Preserve compatibility with old config.pkl files.
    # --> Old ones labelled config.focalstack as either True or False
    # --> New models don't have the 'focalstack' field, instead they have lfformat=[focalstack/stack]
    if config.lfformat == 'focalstack':
        dataset = get_validation_focal_stack_loader(config, args.seq, transform, args.no_pose)
        print("Loading images as focalstack")
    elif config.lfformat == 'stack':
        dataset = get_validation_stacked_lf_loader(config, args.seq, transform, args.no_pose, is_monocular)
        print("Loading images as stack")
    elif config.lfformat == 'epi':
        dataset = get_validation_epi_loader(config, args.seq, transform, args.no_pose)
        print("Loading images as tiled EPIs")
    else:
        raise TypeError("Unknown light field image format. Should be either focalstack, stack or epi")

    output_channels = len(config.cameras)

    if config.lfformat == 'epi':
        if config.cameras_epi == "vertical":
            disp_encoder = EpiEncoder('vertical', config.tilesize).to(device)
            pose_encoder = RelativeEpiEncoder('vertical', config.tilesize).to(device)
            dispnet_input_channels = 16 + len(config.cameras)  # 16 is the number of output channels of the encoder
            posenet_input_channels = 16 + len(config.cameras)  # 16 is the number of output channels of the encoder
        elif config.cameras_epi == "horizontal":
            disp_encoder = EpiEncoder('horizontal', config.tilesize).to(device)
            pose_encoder = RelativeEpiEncoder('horizontal', config.tilesize).to(device)
            dispnet_input_channels = 16 + len(config.cameras)  # 16 is the number of output channels of the encoder
            posenet_input_channels = 16 + len(config.cameras)  # 16 is the number of output channels of the encoder
        elif config.cameras_epi == "full":
            disp_encoder = EpiEncoder('full', config.tilesize).to(device)
            pose_encoder = RelativeEpiEncoder('full', config.tilesize).to(device)

            if "without_disp_stack" not in config:  # for backwards compatibility
                config.without_disp_stack = False

            if config.without_disp_stack:
                dispnet_input_channels = 32  # 16 is the number of output channels of each encoder
            else:
                dispnet_input_channels = 32 + 5  # 16 is the number of output channels of each encoder, 5 from stack
            posenet_input_channels = 32 + 5  # 16 is the number of output channels of each encoder
        else:
            raise ValueError("Incorrect cameras epi format")
        print("Initialised disp and pose encoders")
    else:
        dispnet_input_channels = posenet_input_channels = dataset[0]['tgt_lf_formatted'].shape[0]

    print(f"[DispNet] Using {dispnet_input_channels} input channels, {output_channels} output channels")
    print(f"[PoseNet] Using {posenet_input_channels} input channels")

    dispnet_weights = torch.load(config.dispnet)['state_dict']
    posenet_weights = torch.load(config.posenet)['state_dict']

    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    if enc == "epi":
        visualize_dispnet_encoder_weights(dispnet_weights)
        visualize_posenet_encoder_weights(posenet_weights)
        plt.show()

    # wts = dispnet_weights['conv1.2.weight'].detach().cpu().numpy()
    # fig, axs = plt.subplots(wts.shape[0], wts.shape[1])
    # for wt_i in range(0, wts.shape[0]):
    #     for wt_j in range(0, wts.shape[1]):
    #         axs[wt_i, wt_j].imshow(wts[wt_i, wt_j, :, :])
    #         axs[wt_i, wt_j].get_xaxis().set_visible(False)
    #         axs[wt_i, wt_j].get_yaxis().set_visible(False)
    # plt.show()

    # Load disp net
    disp_net = DispNetS(in_channels=dispnet_input_channels, out_channels=output_channels, encoder=disp_encoder).to(
        device)
    weights = torch.load(config.dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    # load pose net
    pose_net = PoseNet(in_channels=posenet_input_channels, nb_ref_imgs=args.sequence_length - 1,
                       encoder=pose_encoder).to(device)
    weights = torch.load(config.posenet)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    print("Loaded dispnet and posenet")
    fig_d, axs_d = plt.subplots(5, 8)
    plt.suptitle("Encoded EPI Features for Dispnet")

    fig_p, axs_p = plt.subplots(5, 8)
    plt.suptitle("Encoded EPI Features for Posenet")

    # prediction
    poses = []
    for i, validData in enumerate(dataset):
        print("{:03d}/{:03d}".format(i + 1, len(dataset)), end="\r")

        tgt = validData['tgt_lf'].unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in validData['ref_lfs']]

        if config.lfformat == "epi" and config.cameras_epi == "full":
            tgt_lf_formatted_h = validData['tgt_lf_formatted_h'].unsqueeze(0).to(device)
            tgt_lf_formatted_v = validData['tgt_lf_formatted_v'].unsqueeze(0).to(device)
            ref_lfs_formatted_h = [lf.unsqueeze(0).to(device) for lf in validData['ref_lfs_formatted_h']]
            ref_lfs_formatted_v = [lf.unsqueeze(0).to(device) for lf in validData['ref_lfs_formatted_v']]

            tgt_stack = validData['tgt_stack'].unsqueeze(0).to(device)
            ref_stacks = [lf.unsqueeze(0).to(device) for lf in validData['ref_stacks']]

            # Encode the epi images further
            if config.without_disp_stack:
                # Stacked images should not be concatenated with the encoded EPI images
                tgt_encoded_d = disp_net.encode(tgt_lf_formatted_v, None, tgt_lf_formatted_h)
            else:
                # Stacked images should be concatenated with the encoded EPI images
                tgt_encoded_d = disp_net.encode(tgt_lf_formatted_v, tgt_stack, tgt_lf_formatted_h)

            tgt_encoded_p, ref_encoded_p = pose_net.encode(tgt_lf_formatted_v, tgt_stack,
                                                           ref_lfs_formatted_v, ref_stacks,
                                                           tgt_lf_formatted_h, ref_lfs_formatted_h)
        else:
            tgt_formatted = validData['tgt_lf_formatted'].unsqueeze(0).to(device)

            ref_formatted = [r.unsqueeze(0).to(device) for r in validData['ref_lfs_formatted']]

            if disp_net.has_encoder():
                # This will only be called for epi with horizontal or vertical only encoding
                if config.without_disp_stack:
                    # Stacked images should not be concatenated with the encoded EPI images
                    tgt_encoded_d = disp_net.encode(tgt_formatted, None)
                else:
                    # Stacked images should be concatenated with the encoded EPI images
                    # NOTE: Here we stack all 17 images, not 5. Here the images missing from the encoding,
                    # are covered in the stack. We are not using this case in the paper at all.
                    tgt_encoded_d = disp_net.encode(tgt_formatted, tgt)
            else:
                # This will be called for focal stack and stack, where there is no encoding
                tgt_encoded_d = tgt_formatted

            if pose_net.has_encoder():
                tgt_encoded_p, ref_encoded_p = pose_net.encode(tgt_formatted, tgt, ref_formatted, ref)
            else:
                tgt_encoded_p = tgt_formatted
                ref_encoded_p = ref_formatted

        disp = disp_net(tgt_encoded_d)
        depth = 1 / disp
        pose = pose_net(tgt_encoded_p, ref_encoded_p)

        if i == 0:
            # plot the encoded inputs to the dispnet and posenet
            encd = tgt_encoded_d.detach().cpu().numpy()[0]
            encp = tgt_encoded_p.detach().cpu().numpy()[0]
            plot_5x8_grid(axs_d, encd)
            plot_5x8_grid(axs_p, encp)

            cv1 = disp_net.out_conv1.detach().cpu().numpy()
            fig1, axs1 = plt.subplots(4, 8)
            for chi in range(0, 32):
                print("ch: {}, min: {}, max: {}".format(chi, np.min(cv1[0, chi, :, :]), np.max(cv1[0, chi, :, :])))
                axs1[chi // 8, chi % 8].imshow(cv1[0, chi, :, :])
                axs1[chi // 8, chi % 8].get_xaxis().set_visible(False)
                axs1[chi // 8, chi % 8].get_yaxis().set_visible(False)
                axs1[chi // 8, chi % 8].set_title("{}   {:.2f}".format(np.min(cv1[0, chi, :, :]), np.max(cv1[0, chi, :, :])))
            plt.subplots_adjust(top=0.90, bottom=0.1, right=0.9, left=0.1,
                                hspace=0.6, wspace=0.1)
            plt.suptitle("Result of downsampling conv1.0 + relu + conv1.2  + relu")
            plt.show()
        poses.append(pose[0, 0, :].detach().cpu().numpy())
        # plt.pause(0.5)

    print("\nok")





if __name__ == "__main__":
    main()