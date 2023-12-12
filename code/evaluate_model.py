#!/usr/bin/env python3

import argparse
from pathlib import Path
import pickle
import pandas as pd
import torch

import swag

from seisbench.data import WaveformDataset
from utils.evaluation import run_eval, calculate_metrics
from utils.utils import MODELS

# TODO: does not work for bedretto data

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
        help="number of evaluations to use for uncertainty estimation (default: 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="output directory (default None)",
    )
    # TODO: atm a swag model needs to be specified even if mc_dropout is used
    parser.add_argument(
        "--uncertainty-method",
        type=str,
        default='swag',
        required=True,
        choices=['swag', 'mc_dropout'],
        help="Decide wether to use MC Dropout or Swag for uncertainty evaluation."
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

    true, pred, snr = run_eval(model, data.test(), batch_size=512, num_workers=24)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for obj, desc in [(true, "true"), (pred, "pred"), (snr, "snr")]:
        with open(output_dir / f"{args.model}_{desc}.pickle", "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    metrics = calculate_metrics(true, pred, snr, 0.5)

    with open(output_dir / f"{args.model}_metrics.pickle", "wb") as f:
        pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

    def metrics_gen():
        for _ in range(args.no_of_swag_evaluations):
            if args.uncertainty_method == 'swag':
                swag_model.sample(1.0)
                true, pred, snr = run_eval(swag_model, data.test(), batch_size=512, num_workers=24)
                metrics = calculate_metrics(true, pred, snr, 0.5)
                yield metrics
            elif args.uncertainty_method == 'mc_dropout':
                true, pred, snr = run_eval(model, data.test(), batch_size=512, num_workers=24)
                metrics = calculate_metrics(true, pred, snr, 0.5)
                yield metrics
            else:
                print('Chosen uncertainty method is not valid!')
                exit()

    # Only have evalution of uncertainty left
    # In case of mc_droput, set dropout to training
    if args.uncertainty_method == 'mc_dropout':
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    df = pd.DataFrame(metrics_gen())
    with open(output_dir / f"{args.model}_swag_metrics.pickle", "wb") as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
