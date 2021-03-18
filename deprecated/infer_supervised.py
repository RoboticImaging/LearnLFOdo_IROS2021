import os
import torch
import custom_transforms
import argparse
import numpy as np
from deprecated.dataloader import SequenceFolder
# from tqdm import tqdm
from lfmodels import LFPoseNet as PoseNet
from utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Pkl file containing training configuration")
parser.add_argument("--seq", required=True, type=str, help="Name of sequence to perform inference on")
parser.add_argument("--use-latest-not-best", action="store_true", help="Use the latest set of weights rather than the best")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    args = parser.parse_args()
    config = load_config(args.config)

    # directory for storing output
    output_dir = os.path.join(config.save_path, "results", args.seq)

    # load configuration from checkpoints
    if args.use_latest_not_best:
        config.posenet = os.path.join(config.save_path, "posenet_checkpoint.pth.tar")
        output_dir = output_dir + "-latest"
    else:
        config.posenet = os.path.join(config.save_path, "posenet_best.pth.tar")

    os.makedirs(output_dir)

    # define transformations
    transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # load dataset loader
    dataset = SequenceFolder(
        config.data, 
        cameras=config.cameras,
        gray=config.gray,
        sequence_length=config.sequence_length, 
        shuffle=False, 
        train=False, 
        transform=transform,
        sequence=args.seq,
    )

    input_channels = dataset[0][1].shape[0]

    # define posenet
    pose_net = PoseNet(in_channels=input_channels, nb_ref_imgs=2, output_exp=False).to(device)
    weights = torch.load(config.posenet)
    pose_net.load_state_dict(weights['state_dict'])
    pose_net.eval()

    # prediction
    poses = []
    for i, (tgt, tgt_lf, ref, ref_lf, k, kinv, pose_gt) in enumerate(dataset):
        print("{:03d}/{:03d}".format(i+1, len(dataset)), end="\r")
        tgt = tgt.unsqueeze(0).to(device)
        ref = [r.unsqueeze(0).to(device) for r in ref]
        tgt_lf = tgt_lf.unsqueeze(0).to(device)
        ref_lf = [r.unsqueeze(0).to(device) for r in ref_lf]
        exp, pose = pose_net(tgt_lf, ref_lf)
        poses.append(pose[0,1,:].cpu().numpy())

    # save poses
    outdir = os.path.join(output_dir, "poses.npy")
    np.save(outdir, poses)
    print("\nok")


if __name__ == '__main__':
    main()
