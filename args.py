import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('PDFNet_swinB training script', add_help=False)
    parser.add_argument('--COPY', default=True, type=bool)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='Number of steps to accumulate gradients when updating parameters, set to 1 to disable this feature')
    parser.add_argument('--update_half', default=False, type=bool,
                        help='update_half')
    # Model parameters
    parser.add_argument('--model', default='PDFNet_swinB', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--back_bone', default='PDFNet_swinB', type=str,
                        help='back_bone (default: swinB)')
    parser.add_argument('--back_bone_channels_stage1', default=128, type=int)
    parser.add_argument('--back_bone_channels_stage2', default=256, type=int)
    parser.add_argument('--back_bone_channels_stage3', default=512, type=int)
    parser.add_argument('--back_bone_channels_stage4', default=1024, type=int)
    parser.add_argument('--emb', default=128, type=int)
    parser.add_argument('--input_size', default=1024, type=int, help='images input size')
    parser.add_argument('--Crop_size', default=1024, type=int, help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.)')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epochs', type=float, default=300, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--cooldown_epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--finetune_epoch', default=0, type=int)
    # Dataset parameters
    parser.add_argument('--data_path', default='DATA/DIS-DATA/', type=str,
                        help='dataset path')
    parser.add_argument('--chached', default=False, type=bool,
                        help='dataset chached')
    
    parser.add_argument('--checkpoints_save_path', default='checkpoints/PDFNet', type=str,
                        help='path where to save')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval', default=True, type=bool,
                        help='Do evaluation epoch once after training')
    
    parser.add_argument('--eval_metric', default='F1', type=str,help='F1 or MAE')

    parser.add_argument('--DEBUG', default=False, type=bool,
                        help='DEBUG MODE')
    
    return parser