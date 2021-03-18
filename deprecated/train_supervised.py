from deprecated.dataloader import SequenceFolder
import argparse
import time
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import lfmodels as models

from utils import make_save_path, dump_config
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('name', metavar='NAME', help='experiment name')
parser.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/", help='where to save outputs')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler', help='rotation mode for PoseExpnet : [euler, quat]')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros', help='padding mode for image warping')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH', help='path to pre-trained Exp Pose net model')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output', metavar='N', default=0)
parser.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
parser.add_argument('--gray', action='store_true', help="images are grayscale")

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    save_path = make_save_path(args)
    args.save_path = save_path
    dump_config(save_path, args)
    print('=> Saving checkpoints to {}'.format(save_path))
    torch.manual_seed(args.seed)
    tb_writer = SummaryWriter(save_path)

    # Data preprocessing
    train_transform = valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataloader
    print("=> Fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        gray=args.gray,
        cameras=args.cameras,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )
    
    val_set = SequenceFolder(
        args.data,
        gray=args.gray,
        cameras=args.cameras,
        transform=valid_transform,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        shuffle=False
    )

    print('=> {} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('=> {} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    # Create batch loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Pull first example from dataset to check number of channels
    input_channels = train_set[0][1].shape[0]   
    args.epoch_size = len(train_loader)
    print("=> Using {} input channels, {} total batches".format(input_channels, args.epoch_size))
    
    # create model
    print("=> Creating models")
    pose_exp_net = models.LFPoseNet(in_channels=input_channels, nb_ref_imgs=args.sequence_length - 1, output_exp=args.mask_loss_weight > 0).to(device)

    if args.pretrained_exp_pose:
        print("=> Using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    cudnn.benchmark = True
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    print('=> Setting adam solver')

    optim_params = [
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)


    with open(save_path + "/" + args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, pose_exp_net, optimizer, args.epoch_size, logger, tb_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
        
        # evaluate on validation set
        logger.reset_valid_bar()
        valid_loss = validate(args, val_loader, pose_exp_net, logger, tb_writer)

        if valid_loss < best_error or best_error < 0:
            best_error = valid_loss
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": pose_exp_net.module.state_dict()
            }
            torch.save(checkpoint, save_path + "/" + 'posenet_best.pth.tar')
        torch.save(checkpoint, save_path + "/" + 'posenet_checkpoint.pth.tar')

    logger.epoch_bar.finish()


def train(args, train_loader, pose_exp_net, optimizer, epoch_size, logger, tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    pose_exp_net.train()
    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, intrinsics_inv, pose_gt) in enumerate(train_loader):

        data_time.update(time.time() - end)
        tgt_lf = tgt_lf.to(device)
        ref_lfs = [lf.to(device) for lf in ref_lfs]
        pose_gt = pose_gt.to(device)

        explainability_mask, pose = pose_exp_net(tgt_lf, ref_lfs)
        loss = (pose - pose_gt).abs().mean()
        losses.update(loss.item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        logger.train_bar.update(i+1)
        tb_writer.add_scalar('loss/train', loss, n_iter)

        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

def validate(args, val_loader, pose_exp_net, logger, tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    pose_exp_net.eval()
    end = time.time()
    logger.valid_bar.update(0)

    for i, (tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, intrinsics_inv, pose_gt) in enumerate(val_loader):

        data_time.update(time.time() - end)
        tgt_lf = tgt_lf.to(device)
        ref_lfs = [lf.to(device) for lf in ref_lfs]
        pose_gt = pose_gt.to(device)

        explainability_mask, pose = pose_exp_net(tgt_lf, ref_lfs)
        loss = (pose - pose_gt).abs().mean()
        losses.update(loss.item(), args.batch_size)

        batch_time.update(time.time() - end)
        logger.valid_bar.update(i+1)

        if i % args.print_freq == 0:
            logger.valid_writer.write('Validate: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))


        n_iter += 1

    tb_writer.add_scalar('loss/valid', losses.avg[0], n_iter)
    return losses.avg[0]



if __name__ == '__main__':
    main()
