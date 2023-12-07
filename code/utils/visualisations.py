import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc
from .evaluation import calculate_metrics

"""Module for defining various visualisations.

Each function should have one required parameter `metrics`, a dictionary as returned by eval in evalutations.py and saved to a pickle file in evaluate_model.py"""


def confusion_matrix(metrics, ax=None, **kwargs):
    confusion_matrix = metrics["det_confusion_matrix"]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=np.array(["Noise", "Earthquake"]),
    )
    disp.plot(ax=ax, **kwargs)


def residual_histogram(metrics, axs=None, **kwargs):
    p_res = metrics["p_res"] / 100
    s_res = metrics["s_res"] / 100
    snr = metrics["snr"]

    p_indices = np.abs(p_res) < 1
    s_indices = np.abs(s_res) < 1

    axs[0, 0].hist(p_res[p_indices], bins=50)
    axs[1, 0].hist(s_res[s_indices], bins=50)

    axs[0, 1].set_xscale("log")
    axs[1, 1].set_xscale("log")
    axs[0, 1].set_yscale("log")
    axs[1, 1].set_yscale("log")
    axs[0, 1].scatter(snr[p_indices][::100], p_res[p_indices][::100])
    axs[1, 1].scatter(snr[s_indices][::100], s_res[s_indices][::100])


def residual_ecdf(metrics, axs=None, **kwargs):
    # Put P/S picks on seconds scale
    p_res = metrics["p_res"] / 100
    s_res = metrics["s_res"] / 100

    p_indices = np.abs(p_res) < 1
    s_indices = np.abs(s_res) < 1

    axs[0].set_ylabel("P Picks ECDF")
    axs[0].ecdf(np.abs(p_res[p_indices]))

    axs[1].set_ylabel("S Picks ECDF")
    axs[1].ecdf(np.abs(s_res[s_indices]))


def roc_plot(metrics, ax=None, **kwargs):
    fpr, tpr, threshold = metrics["det_roc"]
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:<.2f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")


desc = {
    "det_precision_score": "Precision",
    "det_recall_score": "Recall",
    "det_f1_score": "F1",
}


def detection_treshold_vs_metric(true, pred, snr, metric_key, ax=None):
    def values():
        for det_treshold in np.linspace(0, 1, num=50):
            yield (
                det_treshold,
                calculate_metrics(true, pred, snr, det_treshold)[metric_key],
            )

    ax.set_xlabel("Detection Treshold")
    ax.set_ylabel(desc[metric_key])
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_ylim(top=1)
    ax.plot(*zip(*values()))


def detection_treshold_vs_prec(*args, **kwargs):
    detection_treshold_vs_metric(*args, metric_key="det_precision_score", **kwargs)


def detection_treshold_vs_det_recall(*args, **kwargs):
    detection_treshold_vs_metric(*args, metric_key="det_recall_score", **kwargs)


def detection_treshold_vs_f1(*args, **kwargs):
    detection_treshold_vs_metric(*args, metric_key="det_f1_score", **kwargs)

def model_comparison(metrics, axs=None, **kwargs):
    classifciation_comp = {
        "Precision": 'det_precision_score',
        "Recall": 'det_f1_score',
        "F1": 'det_f1_score',
    }
    regression_comp = {
        "Mean P": 'p_mu',
        "Mean S": 's_mu',
        "std P": 'p_std',
        "std S": 's_std'
    }

    classification_metrics = {}
    regression_metrics = {}

    tasks = (
        (classifciation_comp, classification_metrics, 'Classification'),
        (regression_comp, regression_metrics, 'Regression')
    )

    for k, v in metrics.items():
        for c, m, _ in tasks:
            m[k] = [abs(v[key]) for key in c.values()]

    max_mean = max([max(v[0], v[1]) for v in regression_metrics.values()])
    max_std = max([max(v[2], v[3]) for v in regression_metrics.values()])

    normalizer = [max_mean, max_mean, max_std, max_std]

    for k,v  in regression_metrics.items():
        for idx in range(len(regression_metrics[k])):
            regression_metrics[k][idx] = v[idx]/normalizer[idx]

    width = 0.25  # the width of the bars
    for idx, (comp, values, name) in enumerate(tasks):
        x = np.arange(len(comp.keys()))  # the label locations
        multiplier = 0

        for attribute, measurement in values.items():
            offset = width * multiplier
            rects = axs[idx].bar(x + offset, measurement, width, label=attribute)
            # axs[0].bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[idx].set_title(name)
        axs[idx].set_xticks(x + width, comp.keys())

        if idx == 0:
            axs[idx].set_ylim(0.98,1)
            axs[idx].legend(ncols=3, bbox_to_anchor=(.52, -.1))
