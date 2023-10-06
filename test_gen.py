import os
import time
import math
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *

def normalize_point_clouds(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/GEN_airplane.pt')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=64)
# Sampling
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument('--seed', type=int, default=9988)
# Evaluation Model Params
parser.add_argument('--eval_model_path', type=str, default='./logs_cls/CLS_2023_08_10__11_10_24')
parser.add_argument('--eval_index_name', type=str, default='ClassIndexing.json')
parser.add_argument('--eval_model_name', type=str, default='ckpt_0.000000_10000.pt')

args = parser.parse_args()


# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(args.seed)


cate_iter = args.categories if args.categories != ["all"] else cate_to_list



# Logging
save_dir = os.path.join(args.save_dir, 'GEN_all_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

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


# Loop for checking all categories
# Datasets and loaders
logger.info(f'Loading dataset...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=cate_iter,
    split='test',
    scale_mode=args.normalize,
    seed = args.seed,
    logger=logger,
    even_out=False
)

logger.info(f'[Test] Starting Test...')
ref_pcs = []
ref_cates = []
ids = []
for i, data in enumerate(test_dset.balanced_iter()):
    ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_cates.append(data['cate'])
    ids.append(str(data["id"])+data['cate'])
ref_pcs = torch.cat(ref_pcs, dim=0)

ids_unique = len(list(set(ids))) == len(ids)
logger.info(f"[Test] All IDs unique: {ids_unique}")

# Generate Point Clouds
gen_pcs = []
correct_class_number = 0
for i in tqdm(range(0, math.ceil(len(test_dset) / args.batch_size)), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(z, args.sample_num_points, flexibility=ckpt['args'].flexibility, category_index=cate_to_index(ref_cates[i]))
        gen_pcs.append(x.detach().cpu())
        correct_class_number += cate_eval(x, ref_cates[i], args)
all_class_number = len(gen_pcs) * args.batch_size
gen_pcs = torch.cat(gen_pcs, dim=0)[:len(test_dset)]
if args.normalize is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.normalize, logger=logger)

print(f"Correct Categories: {correct_class_number/all_class_number}")
exit()
# assumption regarding classes: gen_pcs and ref_pcs get compared in the same order and not inverted or anything. 
# I looked and couldnt find any inversion or order change so far.

# Denormalize point clouds, all shapes have zero mean.
# [WARNING]: Do NOT denormalize!
# ref_pcs *= val_dset.stats['std']
# gen_pcs *= val_dset.stats['std']

with torch.no_grad():

    results = {} #compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.batch_size)
    results = {k:v.item() for k, v in results.items()}

    results["category_nn"] = categorize_nn(
        gen_pcs.to(args.device), 
        ref_cates
        )

    print(categorize_nn(ref_pcs.to(args.device), ref_cates))

    results['jsd'] = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    # results["category"] = compute_category_metric(
    #     gen_pcs.to(args.device), 
    #     ref_pcs.to(args.device), 
    #     ref_cates, 
    #     args.batch_size
    #     )

# ref_avg = torch.mean(torch.tensor([results["category"][category]["ref_cate-ref_no_cate"] for category in list(set(ref_cates))]))
# sample_avg = torch.mean(torch.tensor([results["category"][category]["sample-ref_no_cate"] for category in list(set(ref_cates))]))

# logger.info(f'[Test] Coverage  | {results["lgan_cov-CD"]:.6f}')
# logger.info(f'[Test] MinMatDis | {results["lgan_mmd-CD"]:.6f}')
# logger.info(f'[Test] 1NN-Accur | {results["1-NN-CD-acc"]:.6f}')
logger.info(f'[Test] JsnShnDis | {results["jsd"]:.6f}')
logger.info(f'[Test] CateMatch | {results["category_nn"]:.06f}')
logger.info('[Test] Test done.')
