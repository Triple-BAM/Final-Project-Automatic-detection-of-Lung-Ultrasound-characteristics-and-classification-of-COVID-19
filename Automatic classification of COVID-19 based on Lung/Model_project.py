from tensorflow.keras.models import Model
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, \
    plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt


def model_performance(model, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier):
    """
    Confusion matrix and statistics of the classifier performance.
    :param Classifier : trained classifier
    :param y_pred_test, y_pred_proba_test : prediction set
    :param  X_test, y_test: test set
    :return: confusion matrix plot and statistics printed
    """
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
    plt.show()
    Se = recall_score(y_test, y_pred_test, average='micro')
    PPV = precision_score(y_test, y_pred_test, average='micro')
    Acc = accuracy_score(y_test, y_pred_test)
    F1 = f1_score(y_test, y_pred_test, average='micro')
    print(Classifier, ':')
    print('Recall is {:.3f} \nPrecision is {:.3f} \nAccuracy is {:.3f} \n'
          'F1-Score is {:.3f} '.format(Se, PPV, Acc, F1))
    print('AUC is {:.3f}'.format(roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr')))


def Random_forest_classifier(X_train, X_test, y_train, y_test):
    """
    Random Forest Classifier model creation.
    :param X_train, X_test, Y_train, y_test: train-test split data
    :return: rfc (classifier)
             + plots model performance (confusion matrix)
    """
    rfc = RandomForestClassifier(random_state=42, criterion='gini', min_samples_split=12)
    rfc.fit(X_train, y_train)
    y_pred_test = rfc.predict(X_test)
    y_pred_proba_test = rfc.predict_proba(X_test)
    Classifier = 'rfc'
    model_performance(rfc, X_test, y_test, y_pred_test, y_pred_proba_test, Classifier)

    return rfc


class CovidMultiOutputModel():

    """
    Used to generate our multi-output model. This CNN has three branches:
    COVID-19 severity, pleural line regularity, consolidation appearance.S
    Each branch contains a sequence of Convolutional Layers that is defined
    in the make_default_hidden_layers method.
    """


    def __init__(self,
                 input_size,
                 hidden_blocks: int,
                 filters: list,
                 kernel_sizes,  #: list[tuple[int]],
                 paddings: list,
                 activations: list,
                 batchnorms: list,
                 pool_sizes,  #: list[tuple[int]],
                 dropouts: list,
                 covid_params: dict,
                 pleural_params: dict,
                 consolidation_params: dict
                 ):

        """
        :param input_size: Size of input images, e.g. (C,H,W).
        """
        # Assert that all params are the same length.
        assert hidden_blocks == len(filters) == len(kernel_sizes) == len(paddings) == len(activations) == len(batchnorms) == len(pool_sizes) == len(dropouts)

        self.input_size = input_size
        self.hidden_blocks = hidden_blocks
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.activations = activations
        self.batchnorms = batchnorms
        self.pool_sizes = pool_sizes
        self.dropouts = dropouts

        self.covid_params = covid_params  # dense, activation, batch normalization, dropout, final_activation
        self.pleural_params = pleural_params
        self.consolidation_params = consolidation_params



    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers.
        The structure used in this network is defined as:

        Conv3D -> BatchNormalization -> Pooling -> Dropout
        """

        for idx in range(self.hidden_blocks):
            # Conv block idx
            if idx == 0:    # first layer gets external input
                x = Conv3D(filters=self.filters[idx], kernel_size=self.kernel_sizes[idx], padding=self.paddings[idx])(inputs)
            else:
                x = Conv3D(filters=self.filters[idx], kernel_size=self.kernel_sizes[idx], padding=self.paddings[idx])(x)
            x = Activation(self.activations[idx])(x)
            if self.batchnorms[idx]:
                x = BatchNormalization(axis=-1)(x)
            x = MaxPooling3D(pool_size=self.pool_sizes[idx])(x)
            x = Dropout(self.dropouts[idx])(x)

        return x


    def build_covid_severity_branch(self, inputs, num_covid_severity=4):
        """
        Used to build the severity branch of our video classification network.
        This branch is composed of three (default) Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.
        """
        # conv layers
        x = self.make_default_hidden_layers(inputs)

        # covid_params  # dense, activation, batchnorm, dropout, final_activation
        # classifier
        x = Flatten()(x)
        x = Dense(self.covid_params['dense'])(x)   # FC
        x = Activation(self.covid_params['activation'])(x)
        if self.covid_params['batchnorm']:
            x = BatchNormalization()(x)
        x = Dropout(self.covid_params['dropout'])(x)
        x = Dense(num_covid_severity)(x)
        x = Activation(self.covid_params['final_activation'], name="covid_severity_output")(x)

        return x


    def build_pleural_regular_branch(self, inputs, num_pleural_regular=2):
        """
        Used to build the pl-regularity branch of our video classification network.
        This branch is composed of three (default) Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.
        """

        # conv layers
        x = self.make_default_hidden_layers(inputs)

        # classifier
        x = Flatten()(x)
        x = Dense(self.pleural_params['dense'])(x)   # FC
        x = Activation(self.covid_params['activation'])(x)
        if self.pleural_params['batchnorm']:
            x = BatchNormalization()(x)
        x = Dropout(self.pleural_params['dropout'])(x)
        x = Dense(num_pleural_regular)(x)
        x = Activation(self.pleural_params['final_activation'], name="pleural_regular_output")(x)

        return x


    def build_consolidation_branch(self, inputs, num_consolidation = 2):
        """
        Used to build the consolidation branch of our video classification network.
        This branch is composed of three (default) Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.
        """

        # conv layers
        x = self.make_default_hidden_layers(inputs)

        # classifier
        x = Flatten()(x)
        x = Dense(self.consolidation_params['dense'])(x)   # FC
        x = Activation(self.consolidation_params['activation'])(x)
        if self.consolidation_params['batchnorm']:
            x = BatchNormalization()(x)
        x = Dropout(self.consolidation_params['dropout'])(x)
        x = Dense(num_consolidation)(x)
        x = Activation(self.consolidation_params['final_activation'], name="consolidation_output")(x)

        return x

    def assemble_full_model(self, num_covid_severity=4):

        """
        Used to assemble our multi-output model CNN.
        """

        input_shape = self.input_size
        inputs = Input(shape=input_shape)
        covid_severity_branch = self.build_covid_severity_branch(inputs, num_covid_severity)
        pleural_regular_branch = self.build_pleural_regular_branch(inputs)
        consolidation_branch = self.build_consolidation_branch(inputs)

        model = Model(inputs=inputs,
                      outputs=[covid_severity_branch, pleural_regular_branch,
                               consolidation_branch], name="Covid_LUS_net")

        return model
