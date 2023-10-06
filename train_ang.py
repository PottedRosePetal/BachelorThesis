import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_

from utils.misc import get_new_log_dir, get_logger, seed_all, CheckpointManager, BlackHole, log_hyperparams
from models.common import get_linear_scheduler
from models.detector.PointNetClassifier import PointNetAngularClassifier
from utils.initialize import load_iterators, get_args
from utils.augmentation import AugmentData

args = get_args("ang")
seed_all(args.seed)
# Datasets and loaders
augmentator = AugmentData(args)

unique_classes = args.categories
class_to_index = {cls: index for index, cls in enumerate(unique_classes)}

class_to_index["__model_parameters__"] = dict(
    num_points=args.sample_num_points,
    point_dim=3,
    num_disc_angles=augmentator.disc_angles
)

model = PointNetAngularClassifier(**class_to_index["__model_parameters__"]).to(args.device)

class_to_index["__assertion_args__"] = {}
class_to_index["__assertion_args__"]["angle_deadzone"] = args.angle_deadzone
class_to_index["__assertion_args__"]["num_classes"] = len(args.categories)
class_to_index["__assertion_args__"]["num_disc_angles"] = augmentator.disc_angles
class_to_index["__assertion_args__"]["angles"] = [angles.tolist() for angles in augmentator.gen_roll_pitch_yaw(False)]

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='ANG_', postfix='_' + args.tag if args.tag is not None else '')
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

train_iter, val_iter = load_iterators(args, logger, augmentator)

# Define loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
loss_func = torch.nn.CrossEntropyLoss()
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

    optimizer.zero_grad()
    model.train()

    # Forward
    output:torch.Tensor = model(x)
    softmax_output = torch.nn.functional.softmax(output, dim=2).cpu()
    del x, output
    # print(batch["angle"])
    batchwise_angle = augmentator.ind_from_angle_ext(batch["angle"])
    loss = loss_func(softmax_output.view(-1, 4), batchwise_angle.view(-1, 4))

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    # Calculate accuracy
    predicted_class_index = torch.argmax(softmax_output, dim=2).cpu()
    real_class_index = augmentator.ind_from_angle(batch["angle"])
    correct = (predicted_class_index == real_class_index).sum(dim=0)
    total = batchwise_angle.size(0)
    accuracy_x = correct[0].item() / total
    accuracy_y = correct[1].item() / total
    accuracy_z = correct[2].item() / total

    val_accuracy_x, val_accuracy_y, val_accuracy_z = validate()

    avg_accuracy = sum([accuracy_x, accuracy_y, accuracy_z])/3
    avg_val_accuracy = sum([val_accuracy_x, val_accuracy_y, val_accuracy_z])/3
    
    logger.info(f'[Train] Iter {it:04d} | Loss {loss.item():.10f} | Grad {orig_grad_norm:.10f} | Avg Accuracy {avg_accuracy:.4f} | Avg Validation Accuracy {avg_val_accuracy:.4f}')

    writer.add_scalar('train_classifier/loss', loss, it)
    writer.add_scalar('train_classifier/grad', orig_grad_norm, it)
    writer.add_scalar('train_classifier/avg_accuracy', avg_accuracy, it)
    writer.add_scalar('train_classifier/avg_val_accuracy', avg_val_accuracy, it)
    writer.add_scalar('train_classifier_detailed/accuracy_x', accuracy_x, it)
    writer.add_scalar('train_classifier_detailed/accuracy_y', accuracy_y, it)
    writer.add_scalar('train_classifier_detailed/accuracy_z', accuracy_z, it)
    writer.add_scalar('train_classifier_detailed/val_accuracy_x', val_accuracy_x, it)
    writer.add_scalar('train_classifier_detailed/val_accuracy_y', val_accuracy_y, it)
    writer.add_scalar('train_classifier_detailed/val_accuracy_z', val_accuracy_z, it)
    writer.add_scalar('train_classifier/lr', optimizer.param_groups[0]['lr'], it)
    writer.flush()



@torch.no_grad()
def validate():
    model.eval()
    batch = next(val_iter)
    x = batch['pointcloud'].to(args.device)
    predicted_class_index = torch.argmax(torch.nn.functional.softmax(model(x), dim=2), dim=2).cpu()
    real_class_index = augmentator.ind_from_angle(batch["angle"])
    correct = (predicted_class_index == real_class_index).sum(dim=0)
    total = batch["angle"].size(0)
    accuracy_x = correct[0].item() / total
    accuracy_y = correct[1].item() / total
    accuracy_z = correct[2].item() / total
    return accuracy_x, accuracy_y, accuracy_z

#def test():
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i,batch in tqdm(test_iter_reduced):
            x_test = batch['pointcloud'].to(args.device)
            batchwise_class_test = batch["cate"]

            class_index_test = torch.tensor([class_to_index[cls] for cls in batchwise_class_test], dtype=int).to(args.device)

            # Forward
            output_test = model(x_test)
            predicted_class_index_test = torch.argmax(torch.nn.functional.softmax(output_test, dim=1), dim=1)

            # Calculate accuracy for this batch
            total_correct += (predicted_class_index_test == class_index_test).sum().item()
            total_samples += class_index_test.size(0)

            if args.test_size < i:
                break

    test_accuracy = total_correct / total_samples
    return test_accuracy



# Training loop
logger.info('[Train] Start training...')
it = 1
while it <= args.max_iters:
    batch = next(train_iter)
    train(it,batch)
    if not args.disable_val and (it % args.val_freq == 0 or it == args.max_iters):
        opt_states = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
    it += 1