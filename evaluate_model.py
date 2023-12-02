#!/usr/bin/env python3

import pickle
import torch

import swag

from seisbench.data import WaveformDataset
from seisbench.models import EQTransformer
from evaluation import eval


def main():
    # Load STEAD data
    data_path = "/data/STEAD_dataset/.seisbench/datasets/stead/"
    data = WaveformDataset(data_path)

    # Load trained EQTransformer
    model = EQTransformer(in_channels=1)
    checkpoint = torch.load("swag_runs/second swag run/checkpoint-20.pt")
    model.load_state_dict(checkpoint["state_dict"])

    # Load trained EQTransformer+SWAG
    swag_model = swag.posteriors.SWAG(
        EQTransformer,
        no_cov_mat=True,
        max_num_models=20,
        in_channels=1,
    )
    checkpoint = torch.load("swag_runs/second swag run/swag-20.pt")
    swag_model.load_state_dict(checkpoint["state_dict"], strict=False)

    with open('eqt_metrics.pickle', 'wb') as f:
        metrics = eval(model, data.test())
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

    with open('eqt_swag_metrics.pickle', 'wb') as f:
        model.sample(1.0)
        metrics = eval(model, data.test())
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()
