from deprecated.dataloader import get_focal_stack_loaders, get_stacked_lf_loaders
from parser import parse_training_args
# import argparse
import time
import csv
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import lfmodels as models

from utils import tensor2array, save_checkpoint, make_save_path, log_output_tensorboard, dump_config
from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter


def main():
    best_error = -1
    n_iter = 0
    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # parse training arguments
    args = parse_training_args()
    args.training_output_freq = 100         # resetting the training output frequency here.

    # create a folder to save the output of training
    save_path = make_save_path(args)
    args.save_path = save_path
    # save the current configuration to a pickel file
    dump_config(save_path, args)

    print('=> Saving checkpoints to {}'.format(save_path))
    # set manual seed. WHY??
    torch.manual_seed(args.seed)
    # tensorboard summary
    tb_writer = SummaryWriter(save_path)

    # Data preprocessing
    train_transform = valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Load datasets
    print("=> Fetching scenes in '{}'".format(args.data))

    train_set = val_set = None
    if args.lfformat is 'focalstack':
        train_set, val_set = get_focal_stack_loaders(args, train_transform, valid_transform)
    elif args.lfformat is 'stack':
        train_set, val_set = get_stacked_lf_loaders(args, train_transform, valid_transform)

    print('=> {} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('=> {} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    # Create batch loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    # Pull first example from dataset to check number of channels
    input_channels = train_set[0][1].shape[0]   
    args.epoch_size = len(train_loader)
    print("=> Using {} input channels, {} total batches".format(input_channels, args.epoch_size))
    
    # create model
    print("=> Creating models")
    disp_net = models.LFDispNet(in_channels=input_channels).to(torch_device)
    output_exp = args.mask_loss_weight > 0
    pose_exp_net = models.LFPoseNet(in_channels=input_channels,
                                    nb_ref_imgs=args.sequence_length - 1).to(torch_device)

    # Load or initialize weights
    if args.pretrained_exp_pose:
        print("=> Using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_disp:
        print("=> Using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    # Set some torch flags
    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    # Define optimizer
    print('=> Setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr}, 
        {'params': pose_exp_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    # Logging
    with open(os.path.join(save_path, args.log_summary), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(os.path.join(save_path, args.log_full), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    logger = TermLogger(n_epochs=args.epochs,
                        train_size=min(len(train_loader), args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    # train the network
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_exp_net,
                           optimizer, args.epoch_size, logger, tb_writer, n_iter, torch_device)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_exp_net,
                                                  epoch, logger, tb_writer, torch_device)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        # tensorboard logging
        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance,
        # careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_exp_net.module.state_dict()
            }, is_best)

        with open(os.path.join(save_path, args.log_summary), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_exp_net, optimizer,
          epoch_size, logger, tb_writer, n_iter, torch_device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3, w4 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.gt_pose_loss_weight

    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, intrinsics_inv, pose_gt) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(torch_device)
        ref_imgs = [img.to(torch_device) for img in ref_imgs]
        tgt_lf = tgt_lf.to(torch_device)
        ref_lfs = [lf.to(torch_device) for lf in ref_lfs]
        intrinsics = intrinsics.to(torch_device)
        pose_gt = pose_gt.to(torch_device)

        # compute output
        disparities = disp_net(tgt_lf)
        depth = [1/disp for disp in disparities]

        explainability_mask, pose = pose_exp_net(tgt_lf, ref_lfs)

        loss_1, warped, diff = photometric_reconstruction_loss(
            tgt_img, ref_imgs, intrinsics,
            depth, explainability_mask, pose,
            args.rotation_mode, args.padding_mode
        )
        
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth)

        pred_pose_magnitude = pose[:,:,:3].norm(dim=2)
        pose_gt_magnitude = pose_gt[:,:,:3].norm(dim=2)
        pose_loss = (pred_pose_magnitude - pose_gt_magnitude).abs().mean()

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*pose_loss

        if log_losses:
            tb_writer.add_scalar('train/photometric_error', loss_1.item(), n_iter)
            tb_writer.add_scalar('train/smoothness_loss', loss_3.item(), n_iter)
            tb_writer.add_scalar('train/total_loss', loss.item(), n_iter)
            tb_writer.add_scalar('train/pose_loss', pose_loss.item(), n_iter)
            if w2 > 0:
                tb_writer.add_scalar('train/explanability_loss', loss_2.item(), n_iter)

        if log_output:
            tb_writer.add_image('train/Input', tensor2array(tgt_img[0]), n_iter)
            for k, scaled_maps in enumerate(zip(depth, disparities, warped, diff, explainability_mask)):
                log_output_tensorboard(tb_writer, "train", 0, k, n_iter, *scaled_maps)
                break


        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path + "/" + args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_exp_net, epoch,
                        logger, tb_writer, torch_device, sample_nb_to_log=2):
    batch_time = AverageMeter()
    losses = AverageMeter(i=3, precision=4)
    log_outputs = sample_nb_to_log > 0
    w1, w2, w3, w4 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight, args.gt_pose_loss_weight
    poses = np.zeros(((len(val_loader)-1) * args.batch_size * (args.sequence_length-1),6))
    disp_values = np.zeros(((len(val_loader)-1) * args.batch_size * 3))

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, tgt_lf, ref_imgs, ref_lfs, intrinsics, intrinsics_inv, pose_gt) in enumerate(val_loader):
        tgt_img = tgt_img.to(torch_device)
        ref_imgs = [img.to(torch_device) for img in ref_imgs]
        tgt_lf = tgt_lf.to(torch_device)
        ref_lfs = [lf.to(torch_device) for lf in ref_lfs]
        intrinsics = intrinsics.to(torch_device)
        intrinsics_inv = intrinsics_inv.to(torch_device)
        pose_gt = pose_gt.to(torch_device)

        # compute output
        disp = disp_net(tgt_lf)
        depth = 1/disp
        explainability_mask, pose = pose_exp_net(tgt_lf, ref_lfs)

        loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs,
                                                               intrinsics, depth,
                                                               explainability_mask, pose,
                                                               args.rotation_mode, args.padding_mode)
        loss_1 = loss_1.item()
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask).item()
        else:
            loss_2 = 0
        loss_3 = smooth_loss(depth).item()

        pred_pose_magnitude = pose[:,:,:3].norm(dim=2)
        pose_gt_magnitude = pose_gt[:,:,:3].norm(dim=2)
        pose_loss = (pred_pose_magnitude - pose_gt_magnitude).abs().mean()

        if log_outputs and i < sample_nb_to_log - 1:  # log first output of first batches
            if epoch == 0:
                for j,ref in enumerate(ref_imgs):
                    tb_writer.add_image('val/Input {}/{}'.format(j, i), tensor2array(tgt_img[0]), 0)
                    tb_writer.add_image('val/Input {}/{}'.format(j, i), tensor2array(ref[0]), 1)

            log_output_tensorboard(tb_writer, 'val', i, '', epoch, 1./disp, disp, warped[0], diff[0], explainability_mask)

        if log_outputs and i < len(val_loader)-1:
            step = args.batch_size*(args.sequence_length-1)
            poses[i * step:(i+1) * step] = pose.cpu().view(-1,6).numpy()
            step = args.batch_size * 3
            disp_unraveled = disp.cpu().view(args.batch_size, -1)
            disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                            disp_unraveled.median(-1)[0],
                                                            disp_unraveled.max(-1)[0]]).numpy()

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*pose_loss
        losses.update([loss, loss_1, loss_2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))
    if log_outputs:
        prefix = 'valid poses'
        coeffs_names = ['tx', 'ty', 'tz']
        if args.rotation_mode == 'euler':
            coeffs_names.extend(['rx', 'ry', 'rz'])
        elif args.rotation_mode == 'quat':
            coeffs_names.extend(['qx', 'qy', 'qz'])
        for i in range(poses.shape[1]):
            tb_writer.add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses[:,i], epoch)
        tb_writer.add_histogram('disp_values', disp_values, epoch)
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['val/total_loss', 'val/photometric_error', 'val/explainability_loss']


if __name__ == '__main__':
    main()
