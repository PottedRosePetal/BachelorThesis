import math
import torch
from tqdm.auto import tqdm

from utils.dataset import  cate_to_index
from utils.augmentation import AugmentData
from utils.initialize import (
    load_test_iterators, 
    get_args, 
    init_test_logging,
    load_model_eval
    )
from evaluation import (
    compute_all_metrics, 
    jsd_between_point_cloud_sets, 
    cate_eval, 
    angle_eval
    )

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

args = get_args("test")
augmentator = AugmentData(args)
ckpt = torch.load(args.ckpt)
logger = init_test_logging(args)
model = load_model_eval(args, logger, ckpt)
test_iter, dset_size = load_test_iterators(args, logger, augmentator)

ref_pcs = []
ref_cates = []
ref_angles = []

for i, data in enumerate(test_iter):
    if i >= 20:
        break
    ref_pcs.append(data['pointcloud'].cpu())
    ref_cates.append(torch.tensor([cate_to_index(ref_cate) for ref_cate in data['cate']]).unsqueeze(0))
    ref_angles.append(data['angle'])

ref_pcs = torch.cat(ref_pcs, dim=0)
ref_cates = torch.cat(ref_cates, dim=0)

gen_pcs = []
correct_angle_number = []
correct_class_number = 0

for i in tqdm(range(0, math.ceil(len(ref_pcs)/args.test_batch_size)), 'Generate'):
    with torch.no_grad():
        z = torch.randn([args.test_batch_size, ckpt['args'].latent_dim]).to(args.device)
        x = model.sample(
            z,
            args.sample_num_points, 
            flexibility=ckpt['args'].flexibility, 
            category_index=ref_cates[i,:].to(args.device),
            angles=augmentator.ind_from_angle(ref_angles[i])
            )
        correct_class_number += cate_eval(x.to(args.device), ref_cates[i].to(args.device), args, augmentator)
        correct_angle_number.append(angle_eval(x, ref_angles[i], args, augmentator))
        gen_pcs.append(x.detach().cpu())

all_class_number = len(gen_pcs) * args.test_batch_size
gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_batch_size]

if args.scale_mode is not None:
    gen_pcs = normalize_point_clouds(gen_pcs, mode=args.scale_mode, logger=logger)

torch.cuda.empty_cache()
with torch.no_grad():
    results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.test_batch_size)
    results = {k:v.item() for k, v in results.items()}
    results['jsd'] = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results["category"] = correct_class_number/all_class_number
    results["angle"] = sum(correct_angle_number)/len(correct_angle_number)

logger.info(f'[Test] Coverage  | {results["lgan_cov-CD"]:.6f}')
logger.info(f'[Test] MinMatDis | {results["lgan_mmd-CD"]:.6f}')
logger.info(f'[Test] 1NN-Accur | {results["1-NN-CD-acc"]:.6f}')
logger.info(f'[Test] JsnShnDis | {results["jsd"]:.6f}')
logger.info(f'[Test] Cat-Match | {results["category"]:.06f}')
logger.info(f'[Test] Ang-Match | {results["angle"]:.06f}')
logger.info('[Test] Test done.')
