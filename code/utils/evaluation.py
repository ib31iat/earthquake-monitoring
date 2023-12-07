import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torch.utils.data import DataLoader

import seisbench.generate as sbg
from seisbench.data import WaveformDataset
from seisbench.util import worker_seeding

from .augmentations import ChangeChannels
from .utils import predict

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


def get_eval_augmentations():
    p_phases = [key for key, val in phase_dict.items() if val == "P"]
    s_phases = [key for key, val in phase_dict.items() if val == "S"]

    detection_labeller = sbg.DetectionLabeller(
        p_phases, s_phases=s_phases, key=("X", "detections")
    )

    return [
        # sbg.SteeredWindow(windowlen=6000, strategy="pad"),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
        detection_labeller,
        sbg.ChangeDtype(np.float32, "X"),
        sbg.ChangeDtype(np.float32, "y"),
        sbg.ChangeDtype(np.float32, "detections"),
        ChangeChannels(0),
        sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
    ]


def run_eval(
    model,
    data: WaveformDataset,
    batch_size=100,
    num_workers=0,
    detection_threshold: float = 0.5,
    num_predictions: int = 20,
):
    """Evaluate model on data and return a bunch of resulting metrics.

    Keys in result:
    - det_precision_score
    -"""
    print("Start evaluation.")
    print("Load data.")
    data_generator = sbg.GenericGenerator(data)
    data_generator.add_augmentations(get_eval_augmentations())
    data_loader = DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )

    det_true = []
    p_true = []
    s_true = []
    snr = []

    print("Build ground truth.")
    for idx in range(len(data)):
        _, metadata = data.get_sample(idx)
        det = metadata["trace_category"] == "earthquake_local"
        p = metadata["trace_p_arrival_sample"]
        s = metadata["trace_s_arrival_sample"]
        local_snr = metadata["trace_snr_db"]
        if isinstance(local_snr, str):
            local_snr = float(
                local_snr.replace("[", "").replace("]", "").strip().split(" ")[0]
            )
        else:
            local_snr = 0.0

        det_true.append(det)
        p_true.append(p)
        s_true.append(s)
        snr.append(local_snr)

    p_true = np.array(p_true)
    s_true = np.array(s_true)
    snr = np.array(snr)

    print("Calculate predictions.")
    det_preds = []
    p_preds = []
    s_preds = []
    for _ in range(num_predictions):
        predictions = predict(model, data_loader)["predictions"]
        det_pred.append(predictions[:, 0])
        p_pred.append(predictions[:, 1])
        s_pred.append(predictions[:, 2])

    det_preds = np.array(det_preds)
    p_preds = np.array(p_preds)
    s_preds = np.array(s_preds)

    return ((det_true, p_true, s_true), (det_pred, p_pred, s_pred), snr)


def calculate_metrics(true, pred, snr, detection_threshold):
    (det_true, p_true, s_true) = true
    (det_pred, p_pred, s_pred) = pred

    det_roc = roc_curve(det_true, det_pred.copy())

    det_pred = np.ceil(det_pred - detection_threshold)

    # Remove nans (corresponding to noise) and picks of quakes we did not actually detect.
    nans = np.isnan(p_true)
    p_true = p_true[~nans]
    s_true = s_true[~nans]
    p_pred = p_pred[~nans]
    s_pred = s_pred[~nans]
    snr = snr[~nans]

    results = dict()

    results["det_roc"] = det_roc
    results["det_confusion_matrix"] = confusion_matrix(det_true, det_pred)
    for det_metric in [precision_score, recall_score, f1_score]:
        results[f"det_{det_metric.__name__}"] = det_metric(
            det_true, det_pred, zero_division=1
        )

    for pick, true, pred in [("p", p_true, p_pred), ("s", s_true, s_pred)]:
        for name, metric in [("mu", np.mean), ("std", np.std)]:
            results[f"{pick}_{name}"] = metric(true - pred)
        for name, metric in [
            ("MAE", mean_absolute_error),
            ("MAPE", mean_absolute_percentage_error),
            ("RMSE", lambda true, pred: mean_squared_error(true, pred, squared=False)),
        ]:
            results[f"{pick}_{name}"] = metric(true, pred)

    results["p_res"] = p_true - p_pred
    results["s_res"] = s_true - s_pred
    results["snr"] = snr

    return results
