import sys 

from multiwarp_dataloader import get_focal_stack_loaders, get_stacked_lf_loaders, get_epi_loaders
from parser import parse_multiwarp_training_args
import time
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import custom_transforms
import lfmodels as models

from utils import tensor2array, save_checkpoint, save_checkpoint_current, make_save_path, log_output_tensorboard, dump_config
from loss_functions import multiwarp_photometric_loss, explainability_loss, smooth_loss, compute_errors, pose_loss, total_variation_loss
from logger import TermLogger, AverageMeter


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global best_error, n_iter, device
    args = parse_multiwarp_training_args()
    # Some non-optional parameters for training
    args.training_output_freq = 100
    args.tilesize = 8

    save_path = make_save_path(args)
    args.save_path = save_path

    print("Using device: {}".format(device))

    dump_config(save_path, args)
    print('\n\n=> Saving checkpoints to {}'.format(save_path))

    torch.manual_seed(args.seed)                # setting a manual seed for reproducability
    tb_writer = SummaryWriter(save_path)        # tensorboard summary writer

    # Data pre-processing - Just convert arrays to tensor and normalize the data to be largely between 0 and 1
    train_transform = valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create data loader based on the format of the light field
    print("=> Fetching scenes in '{}'".format(args.data))
    train_set, val_set = None, None
    if args.lfformat == 'focalstack':
        train_set, val_set = get_focal_stack_loaders(args, train_transform, valid_transform)
    elif args.lfformat == 'stack':
        is_monocular = False
        if len(args.cameras) == 1 and args.cameras[0] == 8 and args.cameras_stacked == "input":
                is_monocular = True
        train_set, val_set = get_stacked_lf_loaders(args, train_transform, valid_transform, is_monocular=is_monocular)
    elif args.lfformat == 'epi':
        train_set, val_set = get_epi_loaders(args, train_transform, valid_transform)

    print('=> {} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('=> {} samples found in {} validation scenes'.format(len(val_set), len(val_set.scenes)))

    print('=> Multi-warp training, warping {} sub-apertures'.format(len(args.cameras)))

    # Create batch loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    output_channels = len(args.cameras) # for multi-warp photometric loss, we request as many depth values as the cameras used
    args.epoch_size = len(train_loader)
    
    # Create models
    print("=> Creating models")

    if args.lfformat == "epi":
        print("=> Using EPI encoders")
        if args.cameras_epi == "vertical":
            disp_encoder = models.EpiEncoder('vertical', args.tilesize).to(device)
            pose_encoder = models.RelativeEpiEncoder('vertical', args.tilesize).to(device)
            dispnet_input_channels = 16 + len(args.cameras)     # 16 is the number of output channels of the encoder
            posenet_input_channels = 16 + len(args.cameras)     # 16 is the number of output channels of the encoder
        elif args.cameras_epi == "horizontal":
            disp_encoder = models.EpiEncoder('horizontal', args.tilesize).to(device)
            pose_encoder = models.RelativeEpiEncoder('horizontal', args.tilesize).to(device)
            dispnet_input_channels = 16 + len(args.cameras)  # 16 is the number of output channels of the encoder
            posenet_input_channels = 16 + len(args.cameras)  # 16 is the number of output channels of the encoder
        elif args.cameras_epi == "full":
            disp_encoder = models.EpiEncoder('full', args.tilesize).to(device)
            pose_encoder = models.RelativeEpiEncoder('full', args.tilesize).to(device)
            if args.without_disp_stack:
                dispnet_input_channels = 32  # 16 is the number of output channels of each encoder
            else:
                dispnet_input_channels = 32 + 5  # 16 is the number of output channels of each encoder, 5 from stack
            posenet_input_channels = 32 + 5  # 16 is the number of output channels of each encoder
        else:
            raise ValueError("Incorrect cameras epi format")
    else:
        disp_encoder = None
        pose_encoder = None
        # for stack lfformat channels = num_cameras * num_colour_channels
        # for focalstack lfformat channels = num_focal_planes * num_colour_channels
        dispnet_input_channels = posenet_input_channels = train_set[0]['tgt_lf_formatted'].shape[0]
    
    disp_net = models.LFDispNet(in_channels=dispnet_input_channels,
                                out_channels=output_channels, encoder=disp_encoder).to(device)
    pose_net = models.LFPoseNet(in_channels=posenet_input_channels,
                                nb_ref_imgs=args.sequence_length - 1, encoder=pose_encoder).to(device)

    print("=> [DispNet] Using {} input channels, {} output channels".format(dispnet_input_channels, output_channels))
    print("=> [PoseNet] Using {} input channels".format(posenet_input_channels))

    if args.pretrained_exp_pose:
        print("=> [PoseNet] Using pre-trained weights for pose net")
        weights = torch.load(args.pretrained_exp_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        print("=> [PoseNet] training from scratch")
        pose_net.init_weights()

    if args.pretrained_disp:
        print("=> [DispNet] Using pre-trained weights for DispNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        print("=> [DispNet] training from scratch")
        disp_net.init_weights()

    # this flag tells CUDNN to find the optimal set of algorithms for this specific input data size, which improves
    # runtime efficiency, but takes a while to load in the beginning.
    cudnn.benchmark = True
    # disp_net = torch.nn.DataParallel(disp_net)
    # pose_net = torch.nn.DataParallel(pose_net)

    print('=> Setting adam solver')

    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr}, 
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    with open(save_path + "/" + args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(save_path + "/" + args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'pose_loss'])

    logger = TermLogger(n_epochs=args.epochs,
                        train_size=min(len(train_loader), args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    w1 = torch.tensor(args.photo_loss_weight, dtype=torch.float32, device=device, requires_grad=True)
    # w2 = torch.tensor(args.mask_loss_weight, dtype=torch.float32, device=device, requires_grad=True)
    w3 = torch.tensor(args.smooth_loss_weight, dtype=torch.float32, device=device, requires_grad=True)
    # w4 = torch.tensor(args.gt_pose_loss_weight, dtype=torch.float32, device=device, requires_grad=True)

    # add some constant parameters to the log for easy visualization
    tb_writer.add_scalar(tag="batch_size", scalar_value=args.batch_size)

    # tb_writer.add_scalar(tag="mask_loss_weight", scalar_value=args.mask_loss_weight)    # this is not used

    # tb_writer.add_scalar(tag="gt_pose_loss_weight", scalar_value=args.gt_pose_loss_weight)

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, tb_writer, w1, w3)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, tb_writer, w1, w3)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        # update the learning rate (annealing)
        lr_scheduler.step()

        # add the learning rate to the tensorboard logging
        tb_writer.add_scalar(tag="learning_rate", scalar_value=lr_scheduler.get_last_lr()[0], global_step=epoch)

        tb_writer.add_scalar(tag="photometric_loss_weight", scalar_value=w1, global_step=epoch)
        tb_writer.add_scalar(tag="smooth_loss_weight", scalar_value=w3, global_step=epoch)

        # add validation errors to the tensorboard logging
        for error, name in zip(errors, error_names):
            tb_writer.add_scalar(tag=name, scalar_value=error, global_step=epoch)

        decisive_error = errors[2]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(save_path, {'epoch': epoch + 1, 'state_dict': disp_net.state_dict()},
                        {'epoch': epoch + 1, 'state_dict': pose_net.state_dict()}, is_best)

        # save a checkpoint every 20 epochs anyway
        if epoch % 20 == 0:
            save_checkpoint_current(save_path, {'epoch': epoch + 1, 'state_dict': disp_net.state_dict()},
                                    {'epoch': epoch + 1, 'state_dict': pose_net.state_dict()}, epoch)

        with open(save_path + "/" + args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, tb_writer, w1, w3):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    # Set the networks to training mode, batch norm and dropout are handled accordingly
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.start()
    logger.train_bar.update(0)

    for i, trainingdata in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_lf = trainingdata['tgt_lf'].to(device)
        ref_lfs = [img.to(device) for img in trainingdata['ref_lfs']]

        if args.lfformat == "epi" and args.cameras_epi == "full":
            # in this case we have separate horizontal and vertical epis
            tgt_lf_formatted_h = trainingdata['tgt_lf_formatted_h'].to(device)
            tgt_lf_formatted_v = trainingdata['tgt_lf_formatted_v'].to(device)
            ref_lfs_formatted_h = [lf.to(device) for lf in trainingdata['ref_lfs_formatted_h']]
            ref_lfs_formatted_v = [lf.to(device) for lf in trainingdata['ref_lfs_formatted_v']]

            # stacked images
            tgt_stack = trainingdata['tgt_stack'].to(device)
            ref_stacks = [lf.to(device) for lf in trainingdata['ref_stacks']]

            # Encode the epi images further
            if args.without_disp_stack:
                # Stacked images should not be concatenated with the encoded EPI images
                tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted_v, None, tgt_lf_formatted_h)
            else:
                # Stacked images should be concatenated with the encoded EPI images
                tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted_v, tgt_stack, tgt_lf_formatted_h)

            tgt_lf_encoded_p, ref_lfs_encoded_p = pose_net.encode(tgt_lf_formatted_v, tgt_stack,
                                                                  ref_lfs_formatted_v, ref_stacks,
                                                                  tgt_lf_formatted_h, ref_lfs_formatted_h)
        else:
            tgt_lf_formatted = trainingdata['tgt_lf_formatted'].to(device)
            ref_lfs_formatted = [lf.to(device) for lf in trainingdata['ref_lfs_formatted']]

            # Encode the images if necessary
            if disp_net.has_encoder():
                # This will only be called for epi with horizontal or vertical only encoding
                if args.without_disp_stack:
                    # Stacked images should not be concatenated with the encoded EPI images
                    tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted, None)
                else:
                    # Stacked images should be concatenated with the encoded EPI images
                    # NOTE: Here we stack all 17 images, not 5. Here the images missing from the encoding,
                    # are covered in the stack. We are not using this case in the paper at all.
                    tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted, tgt_lf)
            else:
                # This will be called for focal stack and stack, where there is no encoding
                tgt_lf_encoded_d = tgt_lf_formatted

            if pose_net.has_encoder():
                tgt_lf_encoded_p, ref_lfs_encoded_p = pose_net.encode(tgt_lf_formatted, tgt_lf,
                                                                      ref_lfs_formatted, ref_lfs)
            else:
                tgt_lf_encoded_p = tgt_lf_formatted
                ref_lfs_encoded_p = ref_lfs_formatted

        # compute output of networks
        disparities = disp_net(tgt_lf_encoded_d)
        depth = [1/disp for disp in disparities]
        pose = pose_net(tgt_lf_encoded_p, ref_lfs_encoded_p)

        # if i==0:
        #     tb_writer.add_graph(disp_net, tgt_lf_encoded_d)
        #     tb_writer.add_graph(pose_net, (tgt_lf_encoded_p, ref_lfs_encoded_p))

        # compute photometric error
        intrinsics = trainingdata['intrinsics'].to(device)
        pose_gt_tgt_refs = trainingdata['pose_gt_tgt_refs'].to(device)
        metadata = trainingdata['metadata']
        photometric_error, warped, diff = multiwarp_photometric_loss(
            tgt_lf, ref_lfs, intrinsics, depth, pose, metadata, args.rotation_mode, args.padding_mode
        )

        # smoothness_error = smooth_loss(depth)                             # smoothness error
        smoothness_error = total_variation_loss(depth, sum_or_mean="mean")  # total variation error
        # smoothness_error = total_variation_squared_loss(depth)            # total variation error squared version
        mean_distance_error, mean_angle_error = pose_loss(pose, pose_gt_tgt_refs)

        loss = w1 + torch.exp(-1.0 * w1) * photometric_error + w3 + torch.exp(-1.0 * w3) * smoothness_error

        if log_losses:
            tb_writer.add_scalar(tag='train/photometric_error', scalar_value=photometric_error.item(), global_step=n_iter)
            tb_writer.add_scalar(tag='train/smoothness_loss', scalar_value=smoothness_error.item(), global_step=n_iter)
            tb_writer.add_scalar(tag='train/total_loss', scalar_value=loss.item(), global_step=n_iter)
            tb_writer.add_scalar(tag='train/mean_distance_error', scalar_value=mean_distance_error.item(), global_step=n_iter)
            tb_writer.add_scalar(tag='train/mean_angle_error', scalar_value=mean_angle_error.item(), global_step=n_iter)
        if log_output:
            if args.lfformat == "epi" and args.cameras_epi == "full":
                b, n, h, w = tgt_lf_formatted_v.shape
                vis_img = tgt_lf_formatted_v[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5
            else:
                b, n, h, w = tgt_lf_formatted.shape
                vis_img = tgt_lf_formatted[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5

            b, n, h, w = depth[0].shape
            vis_depth = tensor2array(depth[0][0, 0, :, :], colormap='magma')
            vis_disp = tensor2array(disparities[0][0, 0, :, :], colormap='magma')
            vis_enc_f = tgt_lf_encoded_d[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5
            vis_enc_b = tgt_lf_encoded_d[0, -1, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5

            tb_writer.add_image('train/input', vis_img, n_iter)
            tb_writer.add_image('train/encoded_front', vis_enc_f, n_iter)
            tb_writer.add_image('train/encoded_back', vis_enc_b, n_iter)
            tb_writer.add_image('train/depth', vis_depth, n_iter)
            tb_writer.add_image('train/disp', vis_disp, n_iter)

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
            writer.writerow([loss.item(), photometric_error.item(), smoothness_error.item(),
                             mean_distance_error.item(), mean_angle_error.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    logger.train_bar.finish()
    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net,
                        epoch, logger, tb_writer, w1, w3, sample_nb_to_log=2):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = sample_nb_to_log > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.start()
    logger.valid_bar.update(0)
    for i, validdata in enumerate(val_loader):
        tgt_lf = validdata['tgt_lf'].to(device)
        ref_lfs = [ref.to(device) for ref in validdata['ref_lfs']]

        if args.lfformat == "epi" and args.cameras_epi == "full":
            tgt_lf_formatted_h = validdata['tgt_lf_formatted_h'].to(device)
            tgt_lf_formatted_v = validdata['tgt_lf_formatted_v'].to(device)
            ref_lfs_formatted_h = [lf.to(device) for lf in validdata['ref_lfs_formatted_h']]
            ref_lfs_formatted_v = [lf.to(device) for lf in validdata['ref_lfs_formatted_v']]

            tgt_stack = validdata['tgt_stack'].to(device)
            ref_stacks = [lf.to(device) for lf in validdata['ref_stacks']]

            # Encode the epi images further
            if args.without_disp_stack:
                # Stacked images should not be concatenated with the encoded EPI images
                tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted_v, None, tgt_lf_formatted_h)
            else:
                # Stacked images should be concatenated with the encoded EPI images
                tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted_v, tgt_stack, tgt_lf_formatted_h)

            tgt_lf_encoded_p, ref_lfs_encoded_p = pose_net.encode(tgt_lf_formatted_v, tgt_stack,
                                                                  ref_lfs_formatted_v, ref_stacks,
                                                                  tgt_lf_formatted_h, ref_lfs_formatted_h)
        else:
            tgt_lf_formatted = validdata['tgt_lf_formatted'].to(device)
            ref_lfs_formatted = [lf.to(device) for lf in validdata['ref_lfs_formatted']]

            # Encode the images if necessary
            if disp_net.has_encoder():
                # This will only be called for epi with horizontal or vertical only encoding
                if args.without_disp_stack:
                    # Stacked images should not be concatenated with the encoded EPI images
                    tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted, None)
                else:
                    # NOTE: Here we stack all 17 images, not 5. Here the images missing from the encoding,
                    # are covered in the stack. We are not using this case in the paper at all.
                    # Stacked images should be concatenated with the encoded EPI images
                    tgt_lf_encoded_d = disp_net.encode(tgt_lf_formatted, tgt_lf)
            else:
                # This will be called for focal stack and stack, where there is no encoding
                tgt_lf_encoded_d = tgt_lf_formatted

            if pose_net.has_encoder():
                tgt_lf_encoded_p, ref_lfs_encoded_p = pose_net.encode(tgt_lf_formatted, tgt_lf, ref_lfs_formatted, ref_lfs)
            else:
                tgt_lf_encoded_p = tgt_lf_formatted
                ref_lfs_encoded_p = ref_lfs_formatted
        
        # compute output
        disp = disp_net(tgt_lf_encoded_d)
        depth = 1/disp
        pose = pose_net(tgt_lf_encoded_p, ref_lfs_encoded_p)

        # compute photometric error
        intrinsics = validdata['intrinsics'].to(device)
        pose_gt_tgt_refs = validdata['pose_gt_tgt_refs'].to(device)
        metadata = validdata['metadata']
        photometric_error, warped, diff = multiwarp_photometric_loss(
            tgt_lf, ref_lfs, intrinsics, depth, pose, metadata, args.rotation_mode, args.padding_mode
        )

        photometric_error = photometric_error.item()                      # Photometric loss
        # smoothness_error = smooth_loss(depth).item()                      # Smoothness loss
        smoothness_error = total_variation_loss(depth, sum_or_mean="mean").item()   # Total variation loss
        # smoothness_error = total_variation_squared_loss(depth).item()             # Total variation loss squared version
        mean_distance_error, mean_angle_error = pose_loss(pose, pose_gt_tgt_refs).item()                      # Pose loss

        if log_outputs and i < sample_nb_to_log - 1:  # log first output of first batches
            if args.lfformat == "epi" and args.cameras_epi == "full":
                b, n, h, w = tgt_lf_formatted_v.shape
                vis_img = tgt_lf_formatted_v[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5
            else:
                b, n, h, w = tgt_lf_formatted.shape
                vis_img = tgt_lf_formatted[0, 0, :, :].detach().cpu().numpy().reshape(1, h, w) * 0.5 + 0.5

            vis_depth = tensor2array(depth[0, 0, :, :], colormap='magma')
            vis_disp = tensor2array(disp[0, 0, :, :], colormap='magma')

            tb_writer.add_image('val/target_image', vis_img, epoch)
            tb_writer.add_image('val/disp', vis_disp, epoch)
            tb_writer.add_image('val/depth', vis_depth, epoch)

        loss = w1 + torch.exp(-1.0 * w1) * photometric_error + w3 + torch.exp(-1.0 * w3) * smoothness_error
        losses.update([loss, photometric_error, mean_distance_error, mean_angle_error])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['val/total_loss', 'val/photometric_error', 'val/pose_error']


if __name__ == '__main__':
    main()
