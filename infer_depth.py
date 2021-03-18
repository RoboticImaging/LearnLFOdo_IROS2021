import os
import torch
import custom_transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from multiwarp_dataloader import get_validation_focal_stack_loader, get_validation_stacked_lf_loader, get_validation_epi_loader
from lfmodels import LFDispNet as DispNetS
from lfmodels import LFPoseNet as PoseNet
from lfmodels import EpiEncoder, RelativeEpiEncoder
from loss_functions import multiwarp_photometric_loss
from utils import load_config
import sys
import cv2

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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    args = parser.parse_args()
    config = load_config(os.path.join(args.config_dir, "config.pkl"))
    is_monocular = False

    # add missing param to multiwarp stack run
    # if "multiwarp-5" in config.name:
    #     if "/stack" in config.name:
    #         config.cameras_stacked="input"

    # multiwarp-5
    if "multiwarp-5" in config.name or "multiwarp-outer" in config.name or "multiwarp-all" in config.name:
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

    ## directory for storing output
    # config.save_path = os.path.join("/home/dtejaswi/Desktop/joseph_daniel/ral/", mode, enc)
    # output_dir = os.path.join(config.save_path, "results", args.seq)
    output_dir = os.path.join(args.config_dir, "results", args.suffix + args.seq)

    if args.use_checkpoint_at is not None:
        config.dispnet = os.path.join(args.config_dir, "dispnet_" + args.use_checkpoint_at + "_checkpoint.pth.tar")
        output_dir = output_dir + "_epoch_" + args.use_checkpoint_at
    else:
        # load configuration from checkpoints
        if args.use_latest_not_best:
            config.dispnet = os.path.join(args.config_dir, "dispnet_checkpoint.pth.tar")
            output_dir = output_dir + "-latest"
        else:
            config.dispnet = os.path.join(args.config_dir, "dispnet_best.pth.tar")

    os.makedirs(output_dir)
    os.makedirs(output_dir + "/disp")

    # define transformations
    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Load validation dataset loaders
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
            dispnet_input_channels = 16 + len(config.cameras)     # 16 is the number of output channels of the encoder
        elif config.cameras_epi == "horizontal":
            disp_encoder = EpiEncoder('horizontal', config.tilesize).to(device)
            dispnet_input_channels = 16 + len(config.cameras)  # 16 is the number of output channels of the encoder
        elif config.cameras_epi == "full":
            disp_encoder = EpiEncoder('full', config.tilesize).to(device)
            if config.without_disp_stack:
                dispnet_input_channels = 32  # 16 is the number of output channels of each encoder
            else:
                dispnet_input_channels = 32 + 5  # 16 is the number of output channels of each encoder, 5 from stack
        else:
            raise ValueError("Incorrect cameras epi format")
        print("Initialised disp and pose encoders")
    else:
        disp_encoder = None
        dispnet_input_channels = dataset[0]['tgt_lf_formatted'].shape[0]

    print(f"[DispNet] Using {dispnet_input_channels} input channels, {output_channels} output channels")

    # Load disp net
    disp_net = DispNetS(in_channels=dispnet_input_channels, out_channels=output_channels, encoder=disp_encoder).to(device)
    weights = torch.load(config.dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    print("Loaded only dispnet")

    # prediction
    for i, validData in enumerate(dataset):
        print("{:03d}/{:03d}".format(i + 1, len(dataset)), end="\r")

        tgt = validData['tgt_lf'].unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in validData['ref_lfs']]

        if config.lfformat == "epi" and config.cameras_epi =="full":
            tgt_lf_formatted_h = validData['tgt_lf_formatted_h'].unsqueeze(0).to(device)
            tgt_lf_formatted_v = validData['tgt_lf_formatted_v'].unsqueeze(0).to(device)
            tgt_stack = validData['tgt_stack'].unsqueeze(0).to(device)
            # Encode the epi images further
            if config.without_disp_stack:
                # Stacked images should not be concatenated with the encoded EPI images
                tgt_encoded_d = disp_net.encode(tgt_lf_formatted_v, None, tgt_lf_formatted_h)
            else:
                # Stacked images should be concatenated with the encoded EPI images
                tgt_encoded_d = disp_net.encode(tgt_lf_formatted_v, tgt_stack, tgt_lf_formatted_h)
        else:
            tgt_formatted = validData['tgt_lf_formatted'].unsqueeze(0).to(device)
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

        disp = disp_net(tgt_encoded_d)
        # print(output.shape)
        # print(pose.shape)
        # print(tgt.shape)
        # print(ref[0].shape)

        outfile = os.path.join(output_dir, "{:06d}.png".format(i))
        plt.imsave(outfile, tgt.cpu().numpy()[0, 0, :, :], cmap='gray')
        outfile = os.path.join(output_dir, "disp/{:06d}.png".format(i))
        plt.imsave(outfile, disp.cpu().numpy()[0, 0, :, :])
        outfile = os.path.join(output_dir, "disp/{:06d}.npy".format(i))
        np.save(outfile, disp.cpu().numpy()[0, 0, :, :])
    print("\nok")


if __name__ == '__main__':
    main()
