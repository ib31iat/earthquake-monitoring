from sklearn.metrics import ConfusionMatrixDisplay

"""Module for defining various visualisations.

Each function should have one required parameter `metrics`, a dictionary as returned by eval in evalutations.py and saved to a pickle file in evaluate_model.py"""


def confusion_matrix(metrics):
    confusion_matrix = metrics["det_confusion_matrix"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
