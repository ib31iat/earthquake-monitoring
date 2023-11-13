#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy as np
import tabulate
import torch

import torch.nn.functional as F

import seisbench.generate as sbg
from seisbench.data import WaveformDataset
from seisbench.models import EQTransformer
from seisbench.util import worker_seeding
from torch.utils.data import DataLoader

from swag import utils
from swag.posteriors import SWAG

# Argument Parsing
parser = argparse.ArgumentParser(description="SWA Training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="Example",
    help="dataset name (default: Example)",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    required=True,
    help="path to dataset location (default: None)",
)

parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="input batch size (default: 128)",
)

parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    help="model name (default: None)",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="checkpoint to resume training from (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="number of epochs to train (default: 200)",
)

parser.add_argument(
    "--save_freq",
    type=int,
    default=25,
    help="save frequency (default: 25)",
)

parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    help="evaluation frequency (default: 5)",
)

parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    help="initial learning rate (default: 0.01)",
)

parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="SGD momentum (default: 0.9)",
)

parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")

parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    help="SWA start epoch number (default: 161)",
)

parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)

parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)

parser.add_argument(
    "--cov_mat",
    action="store_true",
    help="save sample covariance",
)

parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    help="checkpoint to restor SWA from (default: None)",
)

parser.add_argument(
    "--loss",
    type=str,
    default="CE",
    help="loss to use for training model (default: Cross-entropy)",
)

parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

parser.add_argument(
    "--no_schedule",
    action="store_true",
    help="store schedule",
)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

# FIXME: torch.backend does not exist
# torch.backend.cudnn.benchmark = True
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

print(f"Preparing directory {args.dir}")
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

print(f"Using model {args.model}")
if args.model == "EQTransformer":
    model_cfg = EQTransformer
else:
    # TODO: Better / different error handling
    print("Error.  Only --model=EQTransformer is supported at the moment.")
    sys.exit(1)

print(f"Loading dataset {args.dataset} from {args.dataset_path}")
data = WaveformDataset(args.dataset_path)

print("Preprocessing data")
train, dev, test = data.train_dev_test()

phase_dict = {"trace_p_arrival_sample": "P", "trace_s_arrival_sample": "S"}

train_generator = sbg.GenericGenerator(train)
dev_generator = sbg.GenericGenerator(train)

# TODO: Proper data preprocessing
augmentations = [
    sbg.ChangeDtype(np.float32),
    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
]

train_generator.add_augmentations(augmentations)
dev_generator.add_augmentations(augmentations)

# NOTE: Hard-coded num_workers
# FIXME: Value > 0 gives scray multi-processing error
num_workers = 0

train_loader = DataLoader(
    train_generator,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=worker_seeding,
)
dev_loader = DataLoader(
    dev_generator,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=worker_seeding,
)

print("Preparing model")
# TODO: Pass arguments to EQTransformer
model = model_cfg()
model.to(args.device)

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

if args.swa:
    print("SWAG Training")
    swag_model = SWAG(
        # TODO: Pass arguments to model
        model_cfg,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
    )
    swag_model.to(args.device)
    swa_n = 0
else:
    print("SGD Training")


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd,
)

# TODO: Respect passed argument
if args.loss == "CE":
    loss_fn = F.cross_entropy
else:
    print("Error! Only Cross Entropy Loss (--loss=CE) supported at the moment.")
    sys.exit(2)


def train_epoch(dataloader):
    size = len(dataloader.dataset)
    loss_sum = 0.0

    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(batch["X"].to(model.device))

        # NOTE: pred, the output from EQTransformer, is a 3-tuple; each element is a tensor with dimension (batch_size, sample_size)
        # I stack those three tensors so that the dimensions match up with batch["y"], ie, (batch_size, 3, sample_size) in this case
        pred = torch.stack(pred, dim=1)

        loss = loss_fn(pred, batch["y"].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: Instead of printing, return loss, accuracy, and potentially other measures of interest as return values from function
        # Might need to turn train_loop into a (Python) generator for that to work out nicely, as it potentially prints multiple times per call
        loss, current = loss.item(), batch_id * batch["X"].shape[0]
        # TODO: Unsure why we multiply with batch["X"].size(0) here.
        loss_sum += loss * batch["X"].size(0)
        if batch_id % 5 == 0:
            # TODO: Add args.verbose
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # TODO: Also return some measure of accuracy
    return {"loss": loss_sum / size, "accuracy": None}


def test_loop(dataloader, model=model):
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))

            # NOTE: Transform output from EQTransformer; see comment above
            pred = torch.stack(pred, dim=1)

            test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    # TODO test_loss is averaged over number of batches
    test_loss /= num_batches
    # TODO: Add args.verbose
    print(f"Test avg loss: {test_loss:>8f}\n")
    return {"loss": test_loss, "accuracy": None}


def predict(dataloader):
    # Effectively swa_gaussian/utils.predict
    predictions = []
    targets = []

    model.eval()  # close model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            pred = torch.stack(pred, dim=1)

            predictions.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(batch["y"].numpy())

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


start_epoch = 0
if args.resume is not None:
    print(f"Resume training from {args.resume}")
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if args.swa and (args.swa_resume is not None):
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        model_cfg,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        loading=True,
        # TODO: pass arguments to model
        # TODO: pass keyword arguments to model
    )
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    swa_state_dict=swag_model.state_dict() if args.swa else None,
    swa_n=swa_n if args.swa else None,
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    train_res = train_epoch(train_loader)
    # Evaluate dev set on first epoch, on eval_freq, and on final epoch
    # TODO: What does eval_freq exactly mean
    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        test_res = test_loop(dev_loader)
    else:
        # Make sure test_res is defined properly
        test_res = {"loss": None, "accuracy": None}

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        sgd_res = predict(dev_loader)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("Updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1

        swag_model.collect_model(model)

        # Same check as above for evaluating model
        if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epoch - 1
        ):
            swag_model.sample(0.0)
            # TODO: Need to implement this function ourselves or adapt it so that it works with our EQTransformer interface.
            # NOTE: At some point it might be just easier to reimplement SWAG ourselves; I am doing that partially with train_epoch, test_loop, and predict already anyway.
            # utils.bn_update(dev_loader, swag_model)
            # TODO: evaluating `swag_model' the same way we can evaluate EQTransformer does not work yet.
            # swag_res = test_loop(dev_loader, model=swag_model)
            swag_res = {"loss": None, "accuracy": None}
        else:
            # Ensure swag_res exists
            swag_res = {"loss": None, "accuracy": None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if args.swa:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name="swag",
                state_dict=swag_model.state_dict(),
            )

    time_ep = time.time() - time_ep

    if use_cuda:
        memory_usage = torch.cuda.memory_allocated() / (1024.0**3)
    else:
        memory_usage = None

    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
        memory_usage,
    ]
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]

    # Pretty printing current state of trairing
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

# Save model one more time if `epochs' is not a multiple of `save_freq'
if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    if args.swa and args.epochs > args.swa_start:
        utils.save_checkpoint(
            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
        )

# Save predictions
if args.swa:
    np.savez(
        os.path.join(args.dir, "sgd_ens_preds.npz"),
        predictions=sgd_ens_preds,
        targets=sgd_targets,
    )
