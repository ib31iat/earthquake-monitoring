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

