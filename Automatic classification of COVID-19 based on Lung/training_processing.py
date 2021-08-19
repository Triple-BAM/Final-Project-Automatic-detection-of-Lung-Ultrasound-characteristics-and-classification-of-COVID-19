import matplotlib.pyplot as plt
import plotly.graph_objects as go


def calc_params_train_val(history):
    """
    Plots the metric vs. epoch plots for all parameters in history.history.

    :param history: Training history for CNN containing all relevant scoring metrics.
    :return: Plots (using plotly) of various metric vs. epoch plots.
    """
    # Metric of each feature
    acc_params = ['covid_severity_output_categorical_accuracy',
                  'pleural_regular_output_binary_accuracy',
                  'consolidation_output_binary_accuracy']
    precision_params = ['covid_severity_output_precision',
                        'pleural_regular_output_precision_1',
                        'consolidation_output_precision_2']
    recall_params = ['covid_severity_output_recall',
                     'pleural_regular_output_recall_1',
                     'consolidation_output_recall_2']
    AUC_params = ['covid_severity_output_auc',
                     'pleural_regular_output_auc_1',
                     'consolidation_output_auc_2']
    mae_params = ['covid_severity_output_mae',
                     'pleural_regular_output_mae',
                     'consolidation_output_mae']

    # Accuracy
    for acc_param in acc_params:
        plt.clf()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history[acc_param],
            name='Train'))
        val_acc_param = 'val_' + acc_param
        fig.add_trace(go.Scatter(
            y=history.history[val_acc_param],
            name='Validation'))
        title_acc_param = 'Accuracy for ' + acc_param + ' feature'
        fig.update_layout(height=500,
                          width=700,
                          title=title_acc_param,
                          xaxis_title='Epoch',
                          yaxis_title='Accuracy')
        fig.show()

    # Precision
    for precision_param in precision_params:
        plt.clf()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history[precision_param],
            name='Train'))
        val_precision_param = 'val_' + precision_param
        fig.add_trace(go.Scatter(
            y=history.history[val_precision_param],
            name='Validation'))
        title_precision_param = 'Precision for ' + precision_param + ' feature'
        fig.update_layout(height=500,
                          width=700,
                          title=title_precision_param,
                          xaxis_title='Epoch',
                          yaxis_title='Precision')
        fig.show()

    # Recall
    for recall_param in recall_params:
        plt.clf()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history[recall_param],
            name='Train'))
        val_recall_param = 'val_' + recall_param
        fig.add_trace(go.Scatter(
            y=history.history[val_recall_param],
            name='Validation'))
        title_recall_param = 'Recall for ' + recall_param + ' feature'
        fig.update_layout(height=500,
                          width=700,
                          title=title_recall_param,
                          xaxis_title='Epoch',
                          yaxis_title='Recall')
        fig.show()

    # AUC
    for AUC_param in AUC_params:
        plt.clf()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history[AUC_param],
            name='Train'))
        val_AUC_param = 'val_' + AUC_param
        fig.add_trace(go.Scatter(
            y=history.history[val_AUC_param],
            name='Validation'))
        title_AUC_param = 'AUC for ' + AUC_param + ' feature'
        fig.update_layout(height=500,
                          width=700,
                          title=title_AUC_param,
                          xaxis_title='Epoch',
                          yaxis_title='AUC')
        fig.show()

    # MAE
    for mae_param in mae_params:
        plt.clf()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history[mae_param],
            name='Train'))
        val_mae_param = 'val_' + mae_param
        fig.add_trace(go.Scatter(
            y=history.history[val_mae_param],
            name='Validation'))
        title_mae_param = 'mae for ' + mae_param + ' feature'
        fig.update_layout(height=500,
                          width=700,
                          title=title_mae_param,
                          xaxis_title='Epoch',
                          yaxis_title='mae')
        fig.show()


def calc_loss_train_val(history):
    """
    Plot the total loss vs. epoch plots.

    :param history: Training history for CNN containing all loss metrics.
    :return: Plot (using Plotly) of total loss vs epoch.
    """
    # overall loss
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        y=history.history['loss'],
        name='Train'))
    fig.add_trace(go.Scattergl(
        y=history.history['val_loss'],
        name='Validation'))
    fig.update_layout(height=500,
                      width=700,
                      title='Overall loss',
                      xaxis_title='Epoch',
                      yaxis_title='Loss')
    fig.show()







