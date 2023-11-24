import numpy as np
import torch
from matplotlib import pyplot as plt
from math import ceil

import torch.nn.functional as F

from seisbench.models import EQTransformer
from seisbench.util import worker_seeding
from swag.posteriors import SWAG
import seisbench.generate as sbg
from torch.utils.data import DataLoader
from augmentations import ChangeChannels

"""Separate file for keeping some functions.  Arguably, these could just live in main.py, but this way they should be directly usable in a jupyter notebook via `import utils`."""


def train_epoch(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss_sum = 0.0

    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(batch["X"].to(model.device))

        loss = loss_fn(pred, batch["y"].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch_id * batch["X"].shape[0]

        # TODO: Unsure why we multiply with batch["X"].size(0) here.
        loss_sum += loss * batch["X"].size(0)
        if batch_id % 5 == 0:
            # TODO: Add args.verbose
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # TODO: Also return some measure of accuracy
    return {"loss": loss_sum / size, "accuracy": None}


def test_loop(model, dataloader, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            # HACK: SWAG.device does not work.
            if isinstance(model, SWAG):
                pred = model(batch["X"])
            else:
                pred = model(batch["X"].to(model.device))

            # HACK: See above.
            if isinstance(model, SWAG):
                test_loss += loss_fn(pred, batch["y"]).item()
            else:
                test_loss += loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    # TODO: test_loss is averaged over number of batches
    test_loss /= num_batches
    # TODO: Add args.verbose
    print(f"Test avg loss: {test_loss:>8f}\n")
    return {"loss": test_loss, "accuracy": None}


def predict(model, dataloader):
    """Convenience function for predicting values in `dataloader' using `model'.  Returns a dictionary with keys 'predicitions' and 'targets'"""
    # Effectively swa_gaussian/utils.predict
    predictions = []
    targets = []

    model.eval()  # close model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))

            predictions.append(F.softmax(pred, dim=1).cpu().numpy())
            targets.append(batch["y"].numpy())

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}

def preprocess(data, batch_size):
    """Takes in a WaveformDataset and performs preprocessing on it.  Returns"""
    train, dev, test = data.train_dev_test()

    phase_dict = {"trace_p_arrival_sample": "P", "trace_s_arrival_sample": "S"}

    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)
    test_generator = sbg.GenericGenerator(test)

    # TODO: Proper data preprocessing
    augmentations = [
        sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=3000, windowlen=6000, selection="random", strategy="variable"),
        sbg.RandomWindow(windowlen=6000, strategy="pad"),
        sbg.RandomArrayRotation(keys='X', axis=-1),
        sbg.ChangeDtype(np.float32),
        ChangeChannels(0),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=1e-9, dim=0),
    ]

    train_generator.add_augmentations(augmentations)
    dev_generator.add_augmentations(augmentations)
    test_generator.add_augmentations(augmentations)

    # picks = {}
    # for i in range(0, 6000, 100):
    #     picks[i/100] = 0

    # for idx in range(len(train_generator)):
    #     i = ceil(np.argmax(train_generator[idx]['y'][0])/100)
    #     picks[i] = picks[i] + 1

    # plt.bar(picks.keys(), picks.values())
    # plt.show()

    # NOTE: Hard-coded num_workers
    # FIXME: Value > 0 gives scray multi-processing error
    num_workers = 0

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
