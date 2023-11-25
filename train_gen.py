import math, torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import  cate_to_index, cate_to_list
from utils.augmentation import AugmentData
from models.flow import spectral_norm_power_iteration
from evaluation import compute_all_metrics, jsd_between_point_cloud_sets, cate_eval, angle_eval
from utils.initialize import load_iterators, load_model, init_logging, get_args



# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)

    x = batch['pointcloud'].to(args.device)
    batchwise_class = batch["cate"]
    angles = augmentator.ind_from_angle(batch['angle'])
    
    class_index = torch.tensor([cate_to_list.index(cls) for cls in batchwise_class], dtype=int).to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight, class_index=class_index, angles=angles,writer=writer, it=it) #refers to GaussianVAE.get_loss() or the FlowVAE variant

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f | Learn Rate %.4f' % (
        it, loss.item(), orig_grad_norm, kl_weight, optimizer.param_groups[0]['lr']
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_inspect(it):
    cate_iter = args.categories if args.categories != ["all"] else cate_to_list

    for cate in cate_iter:
        z = torch.randn([args.num_samples_per_cate, args.latent_dim]).to(args.device)
        angles = augmentator.rand_angles(args.num_samples_per_cate)
        if args.iterator == "no_augmentation":
            angles = torch.zeros_like(angles)
        angles = augmentator.ind_from_angle(angles)
        x = model.sample(
                z, 
                args.sample_num_points, 
                flexibility=args.flexibility, 
                category_index=cate_to_index(cate),
                angles=angles
                )
        writer.add_mesh(f'val/pointcloud_{cate}', x, global_step=it)
        logger.info(f'[Inspect] Generating samples for category {cate}...')

    writer.flush()
    logger.info(f'[Inspect] Generation done.')

def test(it):
    global model
    logger.info(f'[Test] Starting Test...')
    ref_pcs = []
    ref_cates = []
    ref_angles = []

    for i, data in enumerate(val_iter):
        if i >= args.test_size / args.val_batch_size:
            break
        ref_pcs.append(data['pointcloud'].cpu())
        ref_cates.append(torch.tensor([cate_to_index(ref_cate) for ref_cate in data['cate']]).unsqueeze(0))
        ref_angles.append(data['angle'])

    ref_pcs = torch.cat(ref_pcs, dim=0)
    ref_cates = torch.cat(ref_cates, dim=0)

    gen_pcs = []
    correct_angle_number = []
    correct_class_number = 0
    for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            x = model.sample(
                z,
                args.sample_num_points, 
                flexibility=args.flexibility, 
                category_index=ref_cates[i,:].to(args.device),
                angles=augmentator.ind_from_angle(ref_angles[i])
                )
            correct_class_number += cate_eval(x.to(args.device), ref_cates[i].to(args.device), args, augmentator)
            correct_angle_number.append(angle_eval(x, ref_angles[i], args, augmentator))
            gen_pcs.append(x.detach().cpu())

    all_class_number = len(gen_pcs) * args.val_batch_size
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]
    
    # assumption regarding classes: gen_pcs and ref_pcs get compared in the same order and not inverted or anything. 
    # I looked and couldnt find any inversion or order change so far.

    # Denormalize point clouds, all shapes have zero mean.
    # [WARNING]: Do NOT denormalize!
    # ref_pcs *= val_dset.stats['std']
    # gen_pcs *= val_dset.stats['std']
    model.cpu()
    torch.cuda.empty_cache()
    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
        results = {k:v.item() for k, v in results.items()}
        results['jsd'] = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results["category"] = correct_class_number/all_class_number
        results["angle"] = sum(correct_angle_number)/len(correct_angle_number)
    
    writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
    writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
    writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
    writer.add_scalar('test/JSD', results['jsd'], global_step=it)
    writer.add_scalar('test/Category_match_percentage', results["category"], global_step=it)
    writer.add_scalar('test/Angle_match_percentage', results["angle"], global_step=it)
    writer.flush()

    
    torch.cuda.empty_cache()
    model = model.to(args.device)

    logger.info(f'[Test] Coverage  | {results["lgan_cov-CD"]:.6f}')
    logger.info(f'[Test] MinMatDis | {results["lgan_mmd-CD"]:.6f}')
    logger.info(f'[Test] 1NN-Accur | {results["1-NN-CD-acc"]:.6f}')
    logger.info(f'[Test] JsnShnDis | {results["jsd"]:.6f}')
    logger.info(f'[Test] Cat-Match | {results["category"]:.06f}')
    logger.info(f'[Test] Ang-Match | {results["angle"]:.06f}')
    logger.info('[Test] Test done.')

if __name__ == "__main__":
    args = get_args()
    augmentator = AugmentData(args)
    logger, ckpt_mgr, writer = init_logging(args)
    train_iter, val_iter = load_iterators(args,logger,augmentator)
    scheduler, optimizer, model = load_model(args,logger)

    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_parameters} parameters")

    roll,pitch,yaw = augmentator.gen_roll_pitch_yaw(False)
    logger.info(f"Data will be augmented with the angles: \n\t{roll}\n\t{pitch}\n\t{yaw}")

    # Main loop
    logger.info('[Train] Start training...')
    it = 1
    initial_test_memory, final_test_memory = 0,0
    while it <= args.max_iters:
        train(it)
        if not args.disable_val and (it % args.val_freq == 0 or it == args.max_iters):
            validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
        if not args.disable_test and (it % args.test_freq == 0 or it == args.max_iters):
            test(it)
        it += 1
