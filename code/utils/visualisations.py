import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc
from .evaluation import calculate_metrics

"""Module for defining various visualisations.

Each function should have one required parameter `metrics`, a dictionary as returned by eval in evalutations.py and saved to a pickle file in evaluate_model.py"""

# Define plotting framework
def set_framework_title(ax, title=None, subtitle=None, title_x=0, y_adjustment=0, subtitle_adjustment=0):
    if subtitle is not None:
        y_title = 1.33-y_adjustment-subtitle_adjustment
        y_subtitle = 1.25-y_adjustment
    else:
        y_title = 1.25-y_adjustment

    title_args = {
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'transform':  ax.transAxes,
        'x': title_x,
        }

    if title is not None:
        ax.text(
            y=y_title,
            s=title,
            fontweight='heavy',
            color='0.2',
            fontsize='large',
            **title_args,
        )

    if subtitle is not None:
        ax.text(
            y=y_subtitle,
            s=subtitle,
            color='0.5',
            fontsize='medium',
            **title_args,
        )

def framework_remove_borders(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('.2')
    ax.tick_params(left = False, which='both')
    ax.tick_params(bottom=True, which='major')
    ax.tick_params(axis='both', labelcolor='.2')

def framework_set_grid(ax):
    ax.yaxis.grid(linewidth=1, color='w')

def framework_apply(ax, grid=True):
    framework_remove_borders(ax)
    if grid:
        framework_set_grid(ax)

def framework_transform_y_labels(ax):
    ax.tick_params(axis='y', labelcolor='.2')
    ticks = ax.get_yticks()
    labels = [f'{x*1e-3:.0f} K' if x >= 1000 else int(x) for x in ticks[:-1]] + ['']

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

def framework_label_y_axis(ax, text, x=0, y_adjustment=0):
    ax.text(
        y=1.1-y_adjustment,
        s=text,
        color='0.2',
        fontsize='medium',
        horizontalalignment='left',
        verticalalignment='top',
        transform= ax.transAxes,
        x=x,
    )

def framework_set_row_description(ax, text):
    ax.set_ylabel(
        text,
        color='.2'
    )
    ax.yaxis.set_label_position("right")

def framework_label_x_axis(ax, text):
    ax.set_xlabel(
        text,
        color='.2'
    )

def framework(ncols=1, nrows=1, title=None, subtitle=None, title_x=0, grid=True, y_size_factor=1, x_size_factor=1, y_adjustment=0, subtitle_adjustment=0):
    # Background color
    facecolor = '.9'

    # Create figure
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    # Set size
    fig.set_size_inches(14*x_size_factor,6*y_size_factor)

    if isinstance(axs, np.ndarray):
        for ax in axs.reshape(-1):
            ax.set_facecolor(facecolor)
            framework_apply(ax, grid)

        set_framework_title(axs.reshape(-1)[0], title, subtitle, title_x, y_adjustment, subtitle_adjustment)
    else:
        axs.set_facecolor(facecolor)
        set_framework_title(axs, title, subtitle, title_x, y_adjustment, subtitle_adjustment)
        framework_apply(axs, grid)

    return fig, axs

def confusion_matrix(metrics, subtitle=None, grid=False, **kwargs):
    fig, ax = framework(1,1, 'Detection Confusion Matrix', subtitle, title_x=-.25 ,grid=grid)
    confusion_matrix = metrics["det_confusion_matrix"]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=np.array(["Noise", "Earthquake"]),
    )
    disp.plot(ax=ax, **kwargs)

def plot_snr_distribution(ax, snr, y, y_indices):
    sample_interval = 100
    # axs[0, 1].set_xlabel("SNR")
    # axs[0, 1].set_ylabel("P Residuals")
    # axs[0, 1].yaxis.set_label_position("right")
    ax.set_xscale("log")
    ax.scatter(
        snr[y_indices][::sample_interval],
        y[y_indices][::sample_interval],
        marker='x',
        color='tab:blue',
        linewidths=1,
    )

def plot_histogram(ax, y, y_indices, n_bins):
    # Color maps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    n, bins, patches = ax.hist(
        y[y_indices],
        bins=n_bins,
    )
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.cool(n[i]/max(n)))

def residual_histogram(metrics, subtitle=None, **kwargs):
    fig, axs = framework(2,2, 'Resdidual Histogram', subtitle, title_x=-.08)
    p_res = metrics["p_res"] / 100
    s_res = metrics["s_res"] / 100
    snr = metrics["snr"]

    p_indices = np.abs(p_res) < 1
    s_indices = np.abs(s_res) < 1

    plot_histogram(axs[0, 0], p_res, p_indices, n_bins=50)
    plot_histogram(axs[1, 0], s_res, s_indices, n_bins=50)
    framework_transform_y_labels(axs[0,0])
    framework_transform_y_labels(axs[1,0])
    framework_label_y_axis(axs[0, 0], 'Number of Residuals', -.08)
    framework_label_x_axis(axs[1, 0], 'Residual')
    axs[0,0].sharex(axs[1,0])

    plot_snr_distribution(axs[0,1], snr, p_res ,p_indices)
    plot_snr_distribution(axs[1,1], snr, s_res, s_indices)
    framework_label_y_axis(axs[0, 1], 'Residual', -.07)
    framework_label_x_axis(axs[1, 1], 'SNR')
    axs[0,1].sharex(axs[1,1])
    axs[0,1].sharey(axs[1,1])

    framework_set_row_description(axs[0,1], 'P-Waves')
    framework_set_row_description(axs[1,1], 'S-Waves')

def plot_ecdf(ax, y, y_indices):
    ax.ecdf(np.abs(y[y_indices]))

def residual_ecdf(metrics, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title='Residual ECDF', subtitle=subtitle, title_x=-.03, y_size_factor=.5, y_adjustment=0.15)
    # Put P/S picks on seconds scale
    p_res = metrics["p_res"] / 100
    s_res = metrics["s_res"] / 100

    p_indices = np.abs(p_res) < 1
    s_indices = np.abs(s_res) < 1

    ax.set_ylim(0,1.1)
    ax.ecdf(np.abs(p_res[p_indices]), label='P Residuals')
    ax.ecdf(np.abs(s_res[s_indices]), label='S Residuals')

    ax.legend(loc='lower right')

def roc_plot(metrics, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title='Residual ECDF', subtitle=subtitle, title_x=-.03, y_size_factor=14/6, y_adjustment=0.19, subtitle_adjustment=0.06)
    fpr, tpr, threshold = metrics["det_roc"]
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:<.2f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    framework_label_x_axis(ax, "False Positive Rate")
    framework_label_y_axis(ax, "True Positive Rate", x=-.03, y_adjustment=0.07)


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
        "Precision": "det_precision_score",
        "Recall": "det_f1_score",
        "F1": "det_f1_score",
    }
    regression_comp = {
        "Mean P": "p_mu",
        "Mean S": "s_mu",
        "std P": "p_std",
        "std S": "s_std",
    }

    classification_metrics = {}
    regression_metrics = {}

    tasks = (
        (classifciation_comp, classification_metrics, "Classification"),
        (regression_comp, regression_metrics, "Regression"),
    )

    for k, v in metrics.items():
        for c, m, _ in tasks:
            m[k] = [abs(v[key]) for key in c.values()]

    max_mean = max([max(v[0], v[1]) for v in regression_metrics.values()])
    max_std = max([max(v[2], v[3]) for v in regression_metrics.values()])

    normalizer = [max_mean, max_mean, max_std, max_std]

    for k, v in regression_metrics.items():
        for idx in range(len(regression_metrics[k])):
            regression_metrics[k][idx] = v[idx] / normalizer[idx]

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
            axs[idx].set_ylim(0.98, 1)
            axs[idx].legend(ncols=3, bbox_to_anchor=(0.52, -0.1))
