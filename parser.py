import argparse


def parse_training_args():
    """
    Parses arguments for training script

    :return: Arguments with which the script was called
    :rtype: Array
    """
    parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields')
    subparsers = parser.add_subparsers(dest="lfformat")
    subparsers.required = True

    # Focal stack training arguments
    focalstack_args = subparsers.add_parser('focalstack', help="Train using focal stacks")
    focalstack_args.add_argument('--num-cameras', type=int, help='how many cameras to use to construct the focal stack')
    focalstack_args.add_argument('--num-planes', type=int, help='how many planes the focal stack should focus on')
    add_common_arguments(focalstack_args)

    # Image-volume training arguments
    stack_args = subparsers.add_parser('stack', help="Train using colour-channel stacks")
    stack_args.add_argument('-c', '--cameras', nargs='+', type=int, help='which cameras to use', default=[8])
    add_common_arguments(stack_args)

    # Parse the args
    args = parser.parse_args()
    return args


def parse_multiwarp_training_args():
    """
    Parses arguments for multi warp training script

    :return: Arguments with which the script was called
    :rtype: argparse object
    """
    parser = argparse.ArgumentParser(description='Unsupervised learning of depth and visual odometry from light fields')
    subparsers = parser.add_subparsers(dest="lfformat")
    subparsers.required = True

    # Focal stack training arguments
    focalstack_args = subparsers.add_parser('focalstack', help="Train using focal stacks")
    focalstack_args.add_argument('-c', '--cameras', nargs='+', type=int, default=[8],
                                 help='which cameras to use in computing the photometric loss via warping')
    focalstack_args.add_argument('--num-cameras', type=int, help='how many cameras to use to construct the focal stack')
    focalstack_args.add_argument('--num-planes', type=int, help='how many planes the focal stack should focus on')
    add_common_arguments(focalstack_args)

    # Colour channel image stack training arguments
    stack_args = subparsers.add_parser('stack', help="Train using colour-channel stacks")
    stack_args.add_argument('-c', '--cameras', nargs='+', type=int, default=[8],
                            help='which cameras to use in computing the photometric loss via warping')
    stack_args.add_argument('-k', '--cameras-stacked', choices=['input', 'full'], type=str, default='input',
                            help='string indicating if the input cameras or all the cameras should be used to '
                                 'generate the stack. either input or full')
    add_common_arguments(stack_args)

    # Epipolar plane image training arguments
    epi_args = subparsers.add_parser('epi', help="Train using epipolar plane images")
    epi_args.add_argument('-c', '--cameras', nargs='+', type=int, default=[8],
                          help='which cameras to use in computing the photometric loss via warping')
    epi_args.add_argument('-e', '--cameras-epi', dest='cameras_epi', choices=['vertical', 'horizontal', 'full'],
                          type=str, default='vertical', help='string indicating which of the cameras should be used to '
                                                             'form the epipolar image. vertical, horizontal or full')
    epi_args.add_argument('--without-disp-stack', action='store_true',
                           help='when this flag is set, the input to dispnet only has the encoded EPI images without'
                                'the stack. If not set then a stack is concatenated')
    add_common_arguments(epi_args)

    args = parser.parse_args()
    return args


def add_common_arguments(subparser):
    """
    Adds common arguments to the input parse

    :param subparser: The parser to which the arguments have to be added
    :type subparser: argparse.ArgumentParser
    """
    # Metadata
    subparser.add_argument('data', metavar='DIR', help='path to dataset')
    subparser.add_argument('name', metavar='NAME', help='experiment name')
    subparser.add_argument('--save-path', metavar='PATH', default="~/Documents/checkpoints/",
                           help='where to save outputs')

    # Training parameters
    subparser.add_argument('--gray', action='store_true',
                           help="images are grayscale")
    subparser.add_argument('--sequence-length', type=int, metavar='N',
                           help='sequence length for training', default=3)
    subparser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                           help='rotation mode for PoseExpnet : [euler, quat]')
    subparser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                           help='padding mode for image warping')
    subparser.add_argument('--epochs', default=200, type=int, metavar='N',
                           help='number of total epochs to run')
    subparser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                           help='mini-batch size')
    subparser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR',
                           help='initial learning rate')
    subparser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                           help='momentum for sgd, alpha parameter for adam')
    subparser.add_argument('--beta', default=0.999, type=float, metavar='M',
                           help='beta parameters for adam')
    subparser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W',
                           help='weight decay')
    subparser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                           help='path to pre-trained dispnet model')
    subparser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                           help='path to pre-trained Exp Pose net model')
    subparser.add_argument('--seed', default=0, type=int,
                           help='seed for random functions, and network initialization')
    subparser.add_argument('-p', '--photo-loss-weight', type=float,
                           help='weight for photometric loss', metavar='W', default=1)
    subparser.add_argument('-m', '--mask-loss-weight', type=float,
                           help='weight for explainabilty mask loss', metavar='W', default=0)
    subparser.add_argument('-s', '--smooth-loss-weight', type=float,
                           help='weight for disparity smoothness loss', metavar='W', default=0.1)
    subparser.add_argument('-g', '--gt-pose-loss-weight', type=float,
                           help='weight for ground truth pose supervision loss', metavar='W', default=0)
    subparser.add_argument('--total-variation', action='store_true',
                           help='when this flag is set, total-variation error is used instead of smoothness loss')
    subparser.add_argument('--forward-backward', action='store_true',
                           help='when this flag is set, an additional forward-backward pose consistency error is'
                                'added to the loss term')
    subparser.add_argument('--fb-loss-weight', type=float,
                           help='weight for forward-backward loss', default=0)

    # Other configurations
    subparser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                           help='number of data loading workers')
    subparser.add_argument('--print-freq', default=10, type=int, metavar='N',
                           help='print frequency')
    subparser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                           help='csv where to save per-epoch train and valid stats')
    subparser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                           help='csv where to save per-gradient descent train stats')
    subparser.add_argument('--log-output', action='store_true',
                           help='will log dispnet outputs and warped imgs at validation step')
    subparser.add_argument('-f', '--training-output-freq', type=int,
                           help='frequency for outputting dispnet outputs and warped imgs at training for all scales if\
                           0 will not output', metavar='N', default=0)
