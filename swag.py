#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy as np
import tabulate
import torch

import torch.nn.functional as F

from seisbench.data import WaveformDataset
from seisbench.models import EQTransformer, EQTransformerReducedEncoder, EQTransformerNoResLSTM

import swag
from swag.posteriors import SWAG

from utils import train_epoch, test_loop, predict, preprocess, make_loss_fn


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
    # TODO: This argument (and consequently, the test dataset) is not being used at all yet.
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
    "--swa_lr",
    type=float,
    default=0.02,
    metavar="LR",
    help="SWA LR (default: 0.02)",
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
    "--num_workers", type=int, default=0, help="Number of Workers (default: 0)"
)

parser.add_argument(
    "--verbose", action="store_true", help="Verbose training)"
)

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
    # TODO: Try pretrained EQTransformer
    model_cfg = EQTransformer
elif args.model == "EQTransformerReducedEncoder":
    model_cfg = EQTransformerReducedEncoder,
elif args.model == "EQTransformerNoResLSTM":
    model_cfg =  EQTransformerNoResLSTM
else:
    # TODO: Better / different error handling
    print("Error.  Only --model=EQTransformer is supported at the moment.")
    sys.exit(1)

print(f"Loading dataset {args.dataset} from {args.dataset_path}")
data = WaveformDataset(args.dataset_path)

print("Preprocessing data")
train_loader, dev_loader, _ = preprocess(data, args.batch_size, args.num_workers)

print("Preparing model")
# TODO: Pass arguments to EQTransformer
model = model_cfg(in_channels=1)
model.to(args.device)
if args.verbose:
    print(model)

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
        in_channels=1,
    )
    swag_model.to(args.device)
    swa_n = 0
else:
    print("SGD Training")


def schedule(epoch):
    """Sets the schedule for current epoch, also depending on whether we do SWA training or not."""
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr_init,
)

# TODO: Respect passed argument
if args.loss == "CE":
    loss_fn = F.binary_cross_entropy
else:
    print("Error! Only Binary Cross Entropy Loss (--loss=CE) supported at the moment.")
    sys.exit(2)

loss_fn = make_loss_fn(loss_fn)


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

columns = [
    "ep",
    "lr",
    "tr_loss",
    "tr_acc",
    "te_loss",
    "te_acc",
    "time",
    "mem_usage",
]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None}

swag.utils.save_checkpoint(
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
        swag.utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    train_res = train_epoch(model, train_loader, loss_fn, optimizer, epoch, verbose=args.verbose)

    if (epoch + 1) % args.save_freq == 0 and args.swa:
        swag.utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
    # Evaluate dev set on first epoch, on eval_freq, and on final epoch
    # TODO: What does eval_freq exactly mean
    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        test_res = test_loop(model, dev_loader, loss_fn, verbose=args.verbose)
    else:
        # Make sure test_res is defined properly
        test_res = {"loss": None, "accuracy": None}

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        sgd_res = predict(model, dev_loader)
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
            or epoch == args.epochs - 1
        ):
            # TODO: Why do the original authors use 0.0 as the argument?
            swag_model.sample(0.0)
            # NOTE: At some point it might be just easier to reimplement SWAG ourselves; I am doing that partially with train_epoch, test_loop, and predict already anyway.
            swag.utils.bn_update(train_loader, swag_model)
            swag_res = test_loop(swag_model, dev_loader, loss_fn, verbose=args.verbose)
        else:
            # Ensure swag_res exists
            swag_res = {"loss": None, "accuracy": None}

        if (epoch + 1) % args.save_freq == 0 and args.swa:
            swag.utils.save_checkpoint(
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
            values = (
                values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
            )

        # Pretty printing current state of trairing
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
        if epoch % 40 == 0 or epoch == args.swa_start:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

# Save model one more time if `epochs' is not a multiple of `save_freq'
if args.epochs % args.save_freq != 0:
    swag.utils.save_checkpoint(
        args.dir,
        epoch + 1,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    if args.swa and args.epochs > args.swa_start:
        swag.utils.save_checkpoint(
            args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
        )

# Save predictions
if args.swa:
    np.savez(
        os.path.join(args.dir, "sgd_ens_preds.npz"),
        predictions=sgd_ens_preds,
        targets=sgd_targets,
    )
