import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


from utils.misc import get_new_log_dir, get_logger, seed_all, CheckpointManager, BlackHole, log_hyperparams
from models.common import get_linear_scheduler
from models.detector.PointNetClassifier import PointNetClassifier
from utils.initialize import load_iterators, get_args
from utils.augmentation import AugmentData


args  = get_args("cls")
seed_all(args.seed)
augmentator = AugmentData(args)

unique_classes = args.categories
class_to_index = {cls: index for index, cls in enumerate(unique_classes)}
net_args = dict(
    num_features=3,
    num_points=args.sample_num_points,
    num_classes=len(args.categories),
    batchnorm=args.batch_norm
)
class_to_index["__model_parameters__"] = net_args

model = PointNetClassifier(**net_args).to(args.device)

class_to_index["__assertion_args__"] = {}
class_to_index["__assertion_args__"]["angle_deadzone"] = args.angle_deadzone
class_to_index["__assertion_args__"]["num_classes"] = len(args.categories)
class_to_index["__assertion_args__"]["num_disc_angles"] = augmentator.disc_angles
class_to_index["__assertion_args__"]["angles"] = [angles.tolist() for angles in augmentator.gen_roll_pitch_yaw(False)]

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='CLS_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer:SummaryWriter = SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
    with open(Path(log_dir).joinpath("ClassIndexing.json"), 'w') as json_file:
        json.dump(class_to_index, json_file, indent=4)
else:
    logger = get_logger('train', None)
    writer:SummaryWriter = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
train_iter, val_iter = load_iterators(args, logger, augmentator)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
loss_func = nn.CrossEntropyLoss()
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

num_parameters = sum(p.numel() for p in model.parameters())
logger.info(f"Model has {num_parameters} parameters, needing {num_parameters*4:,} Bytes of VRAM.")

roll,pitch,yaw = augmentator.gen_roll_pitch_yaw(False)
logger.info(f"Data will be augmented with the angles: \n\t{roll}\n\t{pitch}\n\t{yaw}")

def train(it,batch):
    x = batch['pointcloud'].to(args.device)
    batchwise_class = batch["cate"]

    # Create a mapping from unique class labels to valid class indices
    class_index = torch.tensor([class_to_index[cls] for cls in batchwise_class], dtype=int).to(args.device)

    optimizer.zero_grad()
    model.train()

    # Forward
    output:torch.Tensor = model(x)  # The output is already log probabilities due to F.log_softmax
    del x
    loss = loss_func(output, class_index)  # Use class_index directly for loss calculation

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    # Calculate accuracy
    predicted_class_index = torch.argmax(F.softmax(output, dim=1), dim=1)
    correct = (predicted_class_index == class_index).sum().item()
    total = class_index.size(0)
    accuracy = correct / total
    
    val_acc = validate()
    logger.info(f'[Train] Iter {it:04d} | Loss {loss.item():.10f} | Grad {orig_grad_norm:.4f} | \
Learning Rate {optimizer.param_groups[0]["lr"]:.010f} | Accuracy {accuracy:.7f} | \
Validation Accuracy {val_acc:.7f}')

    writer.add_scalar('train_classifier/loss', loss, it)
    writer.add_scalar('train_classifier/grad', orig_grad_norm, it)
    writer.add_scalar('train_classifier/accuracy', accuracy, it)
    writer.add_scalar('train_classifier/val_accuracy', val_acc, it)
    writer.add_scalar('train_classifier/lr', optimizer.param_groups[0]['lr'], it)
    writer.flush()


@torch.no_grad()
def validate():
    model.eval()
    batch = val_iter.__next__()
    x = batch['pointcloud'].to(args.device)
    batchwise_class = batch["cate"]
    class_index = torch.tensor([class_to_index[cls] for cls in batchwise_class], dtype=int).to(args.device)
    predicted_class_index = torch.argmax(F.softmax(model(x), dim=1), dim=1)
    del x
    correct = (predicted_class_index == class_index).sum().item()
    return correct / class_index.size(0)

# Training loop
logger.info('[Train] Start training...')
it = 1
while it <= args.max_iters:
    batch = train_iter.__next__()
    train(it,batch)
    if not args.disable_val and (it % args.val_freq == 0 or it == args.max_iters):
        opt_states = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
    it += 1