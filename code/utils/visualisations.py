import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc

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
