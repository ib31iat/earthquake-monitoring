import numpy as np
import torch
from matplotlib import pyplot as plt
from math import ceil

import torch.nn.functional as F
from tqdm import tqdm

from seisbench.models import EQTransformer
from seisbench.util import worker_seeding
from swag.posteriors import SWAG
import seisbench.generate as sbg
from torch.utils.data import DataLoader
from .augmentations import ChangeChannels, DuplicateEvent, StoreMetadata

"""Separate file for keeping some functions.  Arguably, these could just live in main.py, but this way they should be directly usable in a jupyter notebook via `import utils`."""


def train_epoch(model, dataloader, loss_fn, optimizer, epoch, verbose=False):
    model.train()
    loss_sum = 0.0
    loss = 0.0

    if verbose:
        pbar = tqdm(dataloader, leave=False)
    else:
        pbar = dataloader

    for batch in pbar:
        # Update progress bar description
        if verbose:
            pbar.set_description(f"epoch: {epoch+1}, loss: {loss:>7f}")
            pbar.refresh()
        # Compute prediction and loss
        pred = model(batch["X"].to(model.device))

        loss = loss_fn(
            pred, batch["y"].to(model.device), batch["detections"].to(model.device)
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()

        loss_sum += loss

    # TODO: Also return some measure of accuracy
    return {"loss": loss_sum, "accuracy": None}


def test_loop(model, dataloader, loss_fn, verbose=False):
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()  # close the model for evaluation

    if verbose:
        pbar = tqdm(dataloader, leave=False)
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch in pbar:
            if isinstance(model, SWAG):
                pred = model(batch["X"].to(model.base.device))
            else:
                pred = model(batch["X"].to(model.device))

            # HACK: See above.
            if isinstance(model, SWAG):
                test_loss += loss_fn(
                    pred,
                    batch["y"].to(model.base.device),
                    batch["detections"].to(model.base.device),
                ).item()
            else:
                test_loss += loss_fn(
                    pred,
                    batch["y"].to(model.device),
                    batch["detections"].to(model.device),
                ).item()

    model.train()  # re-open model for training stage

    # TODO: test_loss is averaged over number of batches
    test_loss /= num_batches

    if verbose:
        print(f"Test avg loss: {test_loss:>8f}\n")
    return {"loss": test_loss, "accuracy": None}


def predict(model, dataloader):
    """Convenience function for predicting values in `dataloader' using `model'.  Returns a dictionary with keys 'predicitions' and 'targets'"""
    predictions = []
    targets = []

    model.eval()  # close model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            # TODO: window_borders does not exist for us.
            # window_borders = batch["window_borders"]

            if isinstance(model, SWAG):
                det_pred, p_pred, s_pred = model(batch["X"].to(model.base.device))

            else:
                det_pred, p_pred, s_pred = model(batch["X"].to(model.device))


            score_detection = torch.zeros(det_pred.shape[0])
            score_p_or_s = torch.zeros(det_pred.shape[0])
            p_sample = torch.zeros(det_pred.shape[0], dtype=int)
            s_sample = torch.zeros(det_pred.shape[0], dtype=int)
            for i in range(det_pred.shape[0]):
                # TODO In pick-benchmark every batch as a "window_borders" property that is used here, but we do not have that (?) so I am just using the full rage instead.
                # start_sample, end_sample = window_borders[i]
                local_det_pred = det_pred[i, :]
                local_p_pred = p_pred[i, :]
                local_s_pred = s_pred[i, :]

                score_detection[i] = torch.max(local_det_pred)
                score_p_or_s[i] = torch.max(local_p_pred) / torch.max(
                    local_s_pred
                )  # most likely P by most likely S

                p_sample[i] = torch.argmax(local_p_pred)
                s_sample[i] = torch.argmax(local_s_pred)

            # TODO Also see notebook for usage
            predictions.append(
                torch.stack((score_detection, p_sample, s_sample), dim=1).cpu()
            )
            targets.append(batch["y"].numpy())

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def split_and_apply_generators(data):
    train, dev, test = data.train_dev_test()
    return {
        "train_generator": sbg.GenericGenerator(train),
        "dev_generator": sbg.GenericGenerator(dev),
        "test_generator": sbg.GenericGenerator(test),
    }


def preprocess(data, batch_size, num_workers):
    """Takes in a WaveformDataset and performs preprocessing on it.  Returns"""
    ################################################################################
    # Configure augmentations; basically: https://github.com/seisbench/pick-benchmark/blob/74ba1965b1dd5e770a8358ed83e339a01460e86b/benchmark/models.py#L440
    ################################################################################

    phase_dict = {
        "trace_p_arrival_sample": "P",
        "trace_pP_arrival_sample": "P",
        "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P",
        "trace_pwPm_arrival_sample": "P",
        "trace_s_arrival_sample": "S",
        "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S",
        "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S",
        "trace_Sn_arrival_sample": "S",
    }

    def get_joint_augmentations(sample_boundaries, sigma):
        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]

        detection_labeller = sbg.DetectionLabeller(
            p_phases, s_phases=s_phases, key=("X", "detections")
        )

        block1 = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=6000,
                        windowlen=12000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=sample_boundaries[0],
                high=sample_boundaries[1],
                windowlen=6000,
                strategy="pad",
            ),
            sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=sigma, dim=0),
            detection_labeller,
            # Normalize to ensure correct augmentation behavior
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block2 = [
            sbg.ChangeDtype(np.float32, "X"),
            sbg.ChangeDtype(np.float32, "y"),
            sbg.ChangeDtype(np.float32, "detections"),
            ChangeChannels(0),
        ]

        return block1, block2

    def get_train_augmentations(rotate_array=False):
        if rotate_array:
            rotation_block = [
                sbg.OneOf(
                    [
                        sbg.RandomArrayRotation(["X", "y", "detections"]),
                        sbg.NullAugmentation(),
                    ],
                    [0.99, 0.01],
                )
            ]
        else:
            rotation_block = []

        augmentation_block = [
            # Add secondary event
            sbg.OneOf(
                [DuplicateEvent(label_keys="y"), sbg.NullAugmentation()],
                probabilities=[0.3, 0.7],
            ),
            # Gaussian noise
            sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [0.5, 0.5]),
            # Array rotation
            *rotation_block,
            # Gaps
            sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [0.2, 0.8]),
            # Channel dropout
            sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [0.3, 0.7]),
            # Augmentations make second normalize necessary
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

        block1, block2 = get_joint_augmentations(
            sample_boundaries=(None, None), sigma=20
        )

        return block1 + augmentation_block + block2

    def get_val_augmentations():
        block1, block2 = get_joint_augmentations(
            sample_boundaries=(None, None), sigma=20
        )

        return block1 + block2

    def get_eval_augmentations():
        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]
        detection_labeller = sbg.DetectionLabeller(
            p_phases, s_phases=s_phases, key=("X", "detections")
        )
        return [
            sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
            detection_labeller,
            StoreMetadata("trace_snr_db"),
            sbg.ChangeDtype(np.float32),
            ChangeChannels(0),
            sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        ]

    ################################################################################
    # Apply augmentations
    ################################################################################

    res = split_and_apply_generators(data)

    train_generator = res["train_generator"]
    dev_generator = res["dev_generator"]
    test_generator = res["test_generator"]

    train_generator.add_augmentations(get_train_augmentations(rotate_array=True))
    dev_generator.add_augmentations(get_val_augmentations())
    test_generator.add_augmentations(get_eval_augmentations())

    # picks = {}
    # for i in range(0, 6000, 100):
    #     picks[i/100] = 0

    # for idx in range(len(train_generator)):
    #     i = ceil(np.argmax(train_generator[idx]['y'][0])/100)
    #     picks[i] = picks[i] + 1

    # plt.bar(picks.keys(), picks.values())
    # plt.show()

    train_loader = DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )
    dev_loader = DataLoader(
        dev_generator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )
    test_loader = DataLoader(
        test_generator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )
    return train_loader, dev_loader, test_loader


def make_loss_fn(loss_fn):
    """Adapes `loss_fn' (e.g. torch.F.cross_entropy or torch.nn.BCELoss) to be compatible with EQTransformer output."""

    def f(pred, y_true, det_true):
        # vector cross entropy loss
        p_true = y_true[:, 0]
        s_true = y_true[:, 1]
        det_true = det_true[:, 0]

        det_pred, p_pred, s_pred = pred

        return (
            0.05 * loss_fn(det_pred.float(), det_true.float())
            + 0.4 * loss_fn(p_pred.float(), p_true.float())
            + 0.55 * loss_fn(s_pred.float(), s_true.float())
        )

    return f
