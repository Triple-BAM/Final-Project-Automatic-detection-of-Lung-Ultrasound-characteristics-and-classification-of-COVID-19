from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def Get_CNN_Confusion_Matrix(y_pred, y_true, param_name):
    """
    Confusion matrix plotting for CNN output.

    :param y_pred: Predicted labels
    :param y_true: GT labels
    :param param_name: Name of the predicted parameter (for plot title)
    :return: Confusion Matrix plot
    """
    # Plot confusion matrix for CNN output
    confusion_mat = confusion_matrix(y_pred, y_true)
    im = ConfusionMatrixDisplay(confusion_mat)
    im.plot()
    plt.title(param_name)
    plt.show()
