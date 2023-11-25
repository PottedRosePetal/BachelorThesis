import argparse

import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import ShapeNetCore
from utils.misc import get_new_log_dir, get_logger, seed_all, CheckpointManager, BlackHole, log_hyperparams, str_list
from utils.data import (
    full_data_iterator, 
    partial_augmentation_iterator,
    full_augmentation_iterator, 
)
from models.vae_gaussian import GaussianVAE
from models.vae_flow import FlowVAE
from models.flow import add_spectral_norm
from models.common import get_linear_scheduler


def get_args(key = "gen"):
    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--model', type=str, default='gaussian', choices=['flow', 'gaussian'])
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='cosine')
    parser.add_argument('--cosine_factor', type=float, default=0.0001)
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=2.0)
    parser.add_argument('--latent_flow_depth', type=int, default=14)
    parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--num_samples_per_cate', type=int, default=2)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--cls_dim', type=int, default=16)      #gets overwritten if only one category is applied
    parser.add_argument('--ang_dim', type=int, default=8)       #gets multiplied by 3, one for each axis
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm_threshold', type=int, default=256) 
        #up to which dimensionality batch norm is applied (inclusive). Higher batch norm can drastically increase runtime.
    parser.add_argument('--extended_residual', type=eval, default=False, choices=[True, False])


    # Datasets and loaders
    parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
    parser.add_argument('--categories', type=str_list, default=['airplane', 'chair'])
    parser.add_argument('--scale_mode', type=str, default='shape_unit')     #global_unit will not work with augmented data
    parser.add_argument('--train_batch_size', type=int, default=192)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--sampler', type=eval, default=True, choices=[True, False])
    parser.add_argument('--even_out', type=eval, default=False, choices=[True, False])  
        # ensures the dataset will have same sizes for cates, at cost of using less of dataset
        # should not be needed anymore, as weighted sampler was introduced
        # DEPRECATED, code for it commented out as of now

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--end_lr', type=float, default=1e-4)
    parser.add_argument('--sched_start_epoch', type=int, default=200_000)
    parser.add_argument('--sched_end_epoch', type=int, default=400_000)

    # Augmentation and Embedding
    parser.add_argument('--iterator', type=str, default='full_augmentation') 
        #choices: "partial_augmentation", "no_augmentation", "full_augmentation"
    parser.add_argument('--aug_multiplier', type=int, default=3)
    parser.add_argument('--jitter_percentage', type=float, default=0.3) #0.3 seems to be optimal, but for comparability I keep 0.05
    parser.add_argument('--aug_iterations', type=int, default=1)
    parser.add_argument('--num_disc_angles', type=int, default=3) # higher means harder to train
    parser.add_argument('--angle_deadzone', type=float, default=torch.pi/2)
    parser.add_argument('--ensure_zero', type=eval, default=True, choices=[True, False])   #may modify num_disc_angles
    parser.add_argument('--check_mem', type=eval, default=False, choices=[True, False])

    parser.add_argument('--deactivate_x', type=eval, default=False, choices=[True, False])
    parser.add_argument('--deactivate_y', type=eval, default=False, choices=[True, False])
    parser.add_argument('--deactivate_z', type=eval, default=False, choices=[True, False])

    # Training
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default=f'./logs_{key}')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=float('inf'))
    parser.add_argument('--val_freq', type=int, default=5000)
    parser.add_argument('--test_freq', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=400)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--disable_test', type=eval, default=False, choices=[True, False])
    parser.add_argument('--disable_val', type=eval, default=False, choices=[True, False])

    # Eval
    # 3 disc angles: CLS_2023_09_04__14_37_56 - 1000; ANG_2023_09_18__17_51_09 - 10000
    # 4 disc angles: CLS_2023_09_02__13_37_36 - 500; ANG_2023_09_01__23_40_12 - 10000 <- prob doesnt work, model change
    parser.add_argument('--cls_eval_model_path', type=str, default='./logs_cls/CLS_2023_09_04__14_37_56')
    parser.add_argument('--cls_eval_index_name', type=str, default='ClassIndexing.json')
    parser.add_argument('--cls_eval_model_name', type=str, default='ckpt_0.000000_1000.pt')

    parser.add_argument('--ang_eval_model_path', type=str, default='./logs_ang/ANG_2023_10_15__23_09_38_no_batchnorm')
    parser.add_argument('--ang_eval_index_name', type=str, default='ClassIndexing.json')
    parser.add_argument('--ang_eval_model_name', type=str, default='ckpt_0.000000_10000.pt')
    parser.add_argument('--spherical_coordinates', type=eval, default=False, choices=[True, False])

    # Test
    parser.add_argument('--ckpt', type=str, default='./pretrained/GEN_airplane.pt')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--test_batch_size', type=int, default=64)


    args = parser.parse_args()

    return args


def init_logging(args):
    seed_all(args.seed)

    # Logging
    if args.logging:
        log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
        logger = get_logger('train', log_dir)
        writer:SummaryWriter = SummaryWriter(log_dir)
        ckpt_mgr = CheckpointManager(log_dir)
        log_hyperparams(writer, args)
    else:
        logger = get_logger('train', None)
        writer:SummaryWriter = BlackHole()
        ckpt_mgr = BlackHole()
    logger.info(args)

    return logger, ckpt_mgr, writer

def init_test_logging(args):
    seed_all(args.seed)

    # Logging
    if args.logging:
        log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
        logger = get_logger('test', log_dir)
        for k, v in vars(args).items():
            logger.info('[ARGS::%s] %s' % (k, repr(v)))
    else:
        logger = get_logger('test', None)
    logger.info(args)

    return logger



def load_iterators(args, logger, augmentator):
    logger.info('Loading datasets...')

    if args.iterator == "full_augmentation":
        iterator = full_augmentation_iterator
    elif args.iterator == "partial_augmentation":
        iterator = partial_augmentation_iterator
    elif args.iterator == "no_augmentation":
        iterator = full_data_iterator
    else:
        raise ValueError("Chosen iterator not Implemented")


    train_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='train',
        scale_mode=args.scale_mode,
        seed = args.seed,
        even_out=args.even_out,
        logger=logger
    )

    if args.sampler:
        train_sampler = train_dset.get_sampler()
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dset,
        batch_size=args.train_batch_size,
        num_workers=0,
        sampler=train_sampler,
        drop_last=True
    )

    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode,
        seed = args.seed,
        even_out=args.even_out,
        logger=logger
    )

    if args.sampler:
        val_sampler = val_dset.get_sampler()
    else:
        val_sampler = None

    val_dataloader = DataLoader(
        val_dset,
        batch_size=args.val_batch_size,
        num_workers=0,
        sampler=val_sampler,
        drop_last=True
    )

    return  iterator(train_dataloader, augmentator), iterator(val_dataloader, augmentator)

def load_test_iterators(args, logger, augmentator):
    logger.info('Loading datasets...')

    if args.iterator == "full_augmentation":
        iterator = full_augmentation_iterator
    elif args.iterator == "partial_augmentation":
        iterator = partial_augmentation_iterator
    elif args.iterator == "no_augmentation":
        iterator = full_data_iterator
    else:
        raise ValueError("Chosen iterator not Implemented")

    test_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=args.scale_mode,
        seed = args.seed,
        even_out=args.even_out,
        logger=logger
    )
    dset_size = len(test_dset)
    logger.info(f"[Test] Test dataset size: {len(test_dset)}")

    if args.sampler:
        test_sampler = test_dset.get_sampler()
    else:
        test_sampler = None

    test_dataloader = DataLoader(
        test_dset,
        batch_size=args.test_batch_size,
        num_workers=0,
        sampler=test_sampler,
        drop_last=True
    )

    return iterator(test_dataloader, augmentator), dset_size


def load_model(args, logger):
    # Model
    logger.info('Building model...')
    if args.model == 'gaussian':
        model = GaussianVAE(args).to(args.device)
    elif args.model == 'flow':
        model = FlowVAE(args).to(args.device)
    logger.debug(repr(model))
    if args.spectral_norm:
        add_spectral_norm(model, logger=logger)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    #decides how the parameters of gaussian noise change in the diffusion process. 
    # In this case it linearly goes down between start and end epoch.
    # basically implemented as learning rate modifier. Applies weight decay on optimizer above. 
    # Needs to be isotropic gaussian, meaning, it has to be gaussian in all dimensions and not prefer one given enough steps.
    # if args.sched_mode == "cosine":
    #     scheduler = get_cosine_scheduler(       
    #         optimizer,
    #         start_epoch=args.sched_start_epoch,
    #         end_epoch=args.sched_end_epoch,
    #         start_lr=args.lr,
    #         end_lr=args.end_lr
    #     )
    # if args.sched_mode == "linear":
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr
    )

    return scheduler, optimizer, model


def load_model_eval(args, logger, ckpt):
    # Model
    logger.info('Loading model...')
    if ckpt['args'].model == 'gaussian':
        model = GaussianVAE(ckpt['args']).to(args.device)
    elif ckpt['args'].model == 'flow':
        model = FlowVAE(ckpt['args']).to(args.device)
    logger.info(repr(model))
    # if ckpt['args'].spectral_norm:
    #     add_spectral_norm(model, logger=logger)
    model.load_state_dict(ckpt['state_dict'])

    return model
