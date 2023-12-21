import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc
from .evaluation import calculate_metrics
import matplotlib.ticker as mticker

"""Module for defining various visualisations.

Each function should have one required parameter `metrics`, a dictionary as returned by eval in evalutations.py and saved to a pickle file in evaluate_model.py"""

COLORS = ['lightsteelblue', 'cornflowerblue', 'royalblue']

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
    ax.spines[['top', 'right', 'left']].set_visible(False)
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

def confusion_matrix(metrics, subtitle=None, grid=False, cmap='YlGn', **kwargs):
    # Color maps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    fig, ax = framework(1,1, 'Detection Confusion Matrix', subtitle, title_x=-.25, y_adjustment=.16,grid=grid, subtitle_adjustment=.04)

    # Remove x-ticks
    ax.tick_params(bottom=False, which='major')

    # Readd spines and change color
    ax.spines[:].set_visible(True)
    ax.spines[:].set_color('.6')


    confusion_matrix = metrics["det_confusion_matrix"]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=np.array(["Noise", "Earthquake"]),
    )
    disp.plot(ax=ax, cmap=cmap, **kwargs)
    return fig

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
        color=plt.cm.autumn_r(0.5),
        linewidths=1,
    )

def plot_histogram(ax, y, y_indices, n_bins):
    # Color maps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    n, bins, patches = ax.hist(
        y[y_indices],
        bins=n_bins,
    )
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.autumn_r(n[i]/max(n)))

def residual_histogram(metrics, subtitle=None, **kwargs):
    fig, axs = framework(2,2, 'Resdidual Histogram', subtitle, title_x=-.1, subtitle_adjustment=-0.02)
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

    framework_set_row_description(axs[0,1], f'P-Waves (hidden: {1-sum(p_indices)/len(p_res):.2f})')
    framework_set_row_description(axs[1,1], f'S-Waves (hidden: {1-sum(s_indices)/len(s_res):.2f})')

    return fig

def plot_ecdf(ax, y, y_indices, color):
    ax.ecdf(np.abs(y[y_indices], color=color))

def residual_ecdf(metrics, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title='Residual ECDF', subtitle=subtitle, title_x=-.03, y_size_factor=.5, y_adjustment=0.1, subtitle_adjustment=-0.02)

    # Put P/S picks on seconds scale
    p_res = metrics["p_res"] / 100
    s_res = metrics["s_res"] / 100

    p_indices = np.abs(p_res) < 1
    s_indices = np.abs(s_res) < 1

    ax.set_ylim(0,1.1)
    ax.ecdf(np.abs(p_res[p_indices]), label=f'P Residuals (hidden: {1-sum(p_indices)/len(p_res):.2f})', color=COLORS[0])
    ax.ecdf(np.abs(s_res[s_indices]), label=f'S Residuals (hidden: {1-sum(s_indices)/len(s_res):.2f})', color=COLORS[2])

    ax.legend(loc='lower right')

    return fig

def roc_plot(metrics, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title='ROC Curve', subtitle=subtitle, title_x=-.03, y_size_factor=14/6, y_adjustment=0.19, subtitle_adjustment=0.06)
    fpr, tpr, threshold = metrics["det_roc"]
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:<.2f}", color=COLORS[2])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    framework_label_x_axis(ax, "False Positive Rate")
    framework_label_y_axis(ax, "True Positive Rate", x=-.03, y_adjustment=0.07)
    ax.legend(loc='center right')

    return fig


desc = {
    "det_precision_score": "Precision",
    "det_recall_score": "Recall",
    "det_f1_score": "F1 Score",
}


def detection_treshold_vs_metric(true, pred, snr, metric_key, ax=None):
    def values():
        for det_treshold in np.linspace(0, 1, num=50):
            yield (
                det_treshold,
                calculate_metrics(true, pred, snr, det_treshold)[metric_key],
            )

    framework_label_x_axis(ax, "Detection Threshold")
    framework_label_y_axis(ax, desc[metric_key], x=-.04, y_adjustment=0.03)

    ax.set_yscale("log")
    ax.yaxis.grid(linewidth=1, color='w', which='both')
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(0, 1)
    ax.set_ylim(top=1)
    ax.plot(*zip(*values()), color=COLORS[2])


def detection_treshold_vs_prec(true, pred, snr, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title=f'Detection Threshold vs. Precision', subtitle=subtitle, title_x=-.04, subtitle_adjustment=0.03, y_adjustment=0.105)
    detection_treshold_vs_metric(true, pred, snr, ax=ax, metric_key="det_precision_score", **kwargs)
    return fig


def detection_treshold_vs_det_recall(true, pred, snr, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title=f'Detection Threshold vs. Recall', subtitle=subtitle, title_x=-.04, subtitle_adjustment=0.03, y_adjustment=0.105)
    detection_treshold_vs_metric(true, pred, snr, ax=ax, metric_key="det_recall_score", **kwargs)
    return fig


def detection_treshold_vs_f1(true, pred, snr, subtitle=None, **kwargs):
    fig, ax = framework(ncols=1, nrows=1, title=f'Detection Threshold vs. F1 Score', subtitle=subtitle, title_x=-.04, subtitle_adjustment=0.03, y_adjustment=0.105)
    detection_treshold_vs_metric(true, pred, snr, ax=ax, metric_key="det_f1_score", **kwargs)
    return fig


def model_comparison(metrics, subtitle=None, **kwargs):
    fig, axs = framework(ncols=3, nrows=1, title=f'EQT Model Comparison', subtitle=subtitle, title_x=-.12, y_adjustment=.05)

    # Build metrics
    classifciation_comp = {
        "Precision": "det_precision_score",
        "Recall": "det_f1_score",
        "F1": "det_f1_score",
    }
    regression_mean = {
        "P-Wave": "p_mu",
        "S-Wave": "s_mu",
    }

    regression_std = {
        "P-Wave": "p_std",
        "S-Wave": "s_std",
    }

    classification_metrics = {}
    regression_mean_metrics = {}
    regression_std_metrics = {}

    tasks = (
        (classifciation_comp, classification_metrics, "Classification"),
        (regression_mean, regression_mean_metrics, "Regression Mean"),
        (regression_std, regression_std_metrics, "Regression Std"),
    )

    for k, v in metrics.items():
        for c, m, _ in tasks:
            m[k] = [abs(v[key]) for key in c.values()]

    max_mean = max([max(v[0], v[1]) for v in regression_mean_metrics.values()])
    max_std = max([max(v[0], v[1]) for v in regression_std_metrics.values()])

    for k, v in regression_mean_metrics.items():
        regression_mean_metrics[k] = v/max_mean

    for k, v in regression_std_metrics.items():
        regression_std_metrics[k] = v/max_std

    width = 0.25  # the width of the bars
    for idx, (comp, values, name) in enumerate(tasks):
        x = np.arange(len(comp.keys()))  # the label locations
        multiplier = 0

        for i, (attribute, measurement) in enumerate(values.items()):
            offset = width * multiplier
            axs[idx].bar(x + offset, measurement, width, label=attribute, color=COLORS[i])
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[idx].set_title(name, color='.2')
        axs[idx].set_xticks(x + width, comp.keys())

        if idx == 0:
            axs[idx].set_ylim(0.98, 1.001)

        if idx == 1:
            axs[idx].legend(ncols=3, bbox_to_anchor=(1.04, 1.18), frameon=False)

    return fig
