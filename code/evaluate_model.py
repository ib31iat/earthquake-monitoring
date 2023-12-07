#!/usr/bin/env python3

import argparse
from pathlib import Path
import pickle
import pandas as pd
import torch

import swag

from seisbench.data import WaveformDataset
from utils.evaluation import eval
from utils.utils import MODELS


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="training directory (default: None)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EQTransformer",
        required=True,
        help="model (default: EQTransformer)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="path to model file (default: None)",
    )
    parser.add_argument(
        "--swag_path",
        type=str,
        default=None,
        required=True,
        help="path to swag file (default: None)",
    )
    parser.add_argument(
        "--no_of_swag_evaluations",
        type=int,
        default=1,
        required=False,
        help="number of evaluations to use for uncertainty estimation (default: 1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="output directory (default None)",
    )
    args = parser.parse_args()

    # Load (entire) dataset
    data = WaveformDataset(args.data_dir)

    # Load trained model from path
    model = MODELS[args.model](in_channels=1)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # Load trained Model+SWAG
    swag_model = swag.posteriors.SWAG(
        MODELS[args.model],
        no_cov_mat=True,
        max_num_models=20,
        in_channels=1,
    )
    checkpoint = torch.load(args.swag_path)
    swag_model.load_state_dict(checkpoint["state_dict"], strict=False)

    metrics = eval(model, data.test(), batch_size=512, num_workers=24)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'eqt_metrics.pickle', 'wb') as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

    def metrics_gen():
        for _ in range(args.no_of_swag_evaluations):
            swag_model.sample(1.0)
            metrics = eval(swag_model, data.test(), batch_size=512, num_workers=24)
            yield metrics

    df = pd.DataFrame(metrics_gen())
    with open(output_dir / 'eqt_swag_metrics.pickle', 'wb') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()
