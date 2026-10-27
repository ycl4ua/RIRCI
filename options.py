import src.networks as networks

model_names = sorted(name for name in networks.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networks.__dict__[name]))


class Options():
    """Command line options shared by training and evaluation."""

    def __init__(self):
        pass

    def init(self, parser):
        # Model structure
        parser.add_argument('--nets', '-n', metavar='NET', default='rirci',
                            choices=model_names,
                            help='network architecture: ' + ' | '.join(model_names))
        parser.add_argument('--models', '-m', metavar='MODEL', default='rirci',
                            help='training wrapper registered in src/models')

        # Training strategy
        parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                            help='number of data loading workers')
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                            help='training batch size')
        parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                            help='test batch size')
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--beta1', default=0.9, type=float, help='Adam beta1')
        parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta2')
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay')
        parser.add_argument('--schedule', type=int, nargs='+', default=[65],
                            help='decrease the learning rate at these epochs')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='learning rate is multiplied by gamma on schedule')

        # Loss weights (the HWVOC / PW setting reported in the paper)
        parser.add_argument('--lambda_l1', type=float, default=2,
                            help='lambda_1: weight of the pixel losses')
        parser.add_argument('--lambda_content', default=1, type=float,
                            help='lambda_2: weight of the VGG perceptual loss')
        parser.add_argument('--lambda_mask', default=3, type=float,
                            help='lambda_3: weight of the mask loss')

        # Data
        parser.add_argument('--dataset', default='rirci', type=str,
                            help='dataset reader registered in datasets/')
        parser.add_argument('--dataset_dir', default='data/RIRCI', type=str,
                            metavar='PATH', help='dataset root (contains train/ and valid/)')
        parser.add_argument('--eval_split', default='val', type=str,
                            choices=['val', 'test'],
                            help='split evaluated by test.py')
        parser.add_argument('--input-size', default=256, type=int, metavar='N',
                            help='network input size')
        parser.add_argument('--crop_size', default=256, type=int)

        # Checkpoints / misc
        parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str,
                            metavar='PATH', help='checkpoint directory')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='checkpoint to resume from (default: none)')
        parser.add_argument('--name', default='rirci', type=str,
                            help='experiment name used as a sub-directory')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate the model instead of training')
        parser.add_argument('--freq', default=-1, type=int, metavar='N',
                            help='validate every N iterations; negative means every epoch')
        parser.add_argument('--gpu_id', default='0', type=str)

        # Architecture switches kept from the code base
        parser.add_argument('--mask_mode', default='res', type=str)  # vanilla, cat, ca, psp, res
        parser.add_argument('--bg_mode', default='res_mask', type=str)  # vanilla, res_mask, res_feat, proposed
        parser.add_argument('--sim_metric', default='cos', type=str)
        parser.add_argument('--project_mode', default='simple', type=str)
        parser.add_argument('--k_center', default=2, type=int,
                            help='1 for foreground only, 2 for background and foreground')
        parser.add_argument('--use_refine', action='store_true')
        parser.add_argument('--gan-norm', default=False, type=bool)
        parser.add_argument('--hl', default=False, type=bool)

        return parser
