# general imports
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from Model_project import CovidMultiOutputModel, model_performance, Random_forest_classifier
from data_parsing import parse_dataset, plot_distribution
from training_processing import calc_loss_train_val, calc_params_train_val
from test_processing import Get_CNN_Confusion_Matrix
from skvideo.io import vread, vwrite
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import cv2
import argparse

parser = argparse.ArgumentParser(description='multi outputs main')
parser.add_argument('--vid_path', type=str,  help='path to videos dir')
parser.add_argument('--metadata_path', type=str,  help='path to GT Excel file')


if __name__ == '__main__':
    # Setting the device to GPU
    config = tf.compat.v1.ConfigProto(gpu_options=
                                      tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                      # device_count = {'GPU': 1}
                                      )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    TF_ENABLE_GPU_GARBAGE_COLLECTION = False

    #  Parsing the data
    args = parser.parse_args()
    vid_path = args.vid_path
    metadata_path = args.metadata_path

    # metadata_path = r'C:\Users\dekel\OneDrive\Final Project\Mor_Deep\Manual_Tags_Final.xls'
    # vid_path = r'C:\Users\dekel\OneDrive\Final Project\Mor_Deep\Masks_Final'
    df = parse_dataset(metadata_path)
    # Plotting data distribution (with plotly)
    plot_distribution(df['covid_severity_grade'])
    plot_distribution(df['pleural_line_regular'])
    plot_distribution(df['consolidation'])
    # Dropping unnecessary column from df
    df = df.drop(['video_name'], axis=1)
    # Creation of initial y matrix
    y = np.asarray(df)

    # Desired frame resolution.
    IM_WIDTH = 128
    IM_HEIGHT = 128

    # Uncomment following code snippet if desired frame resolution is not 200x200.
    # Make a new folder with videos changed from 200x200 to IM_WIDTH x IM_HEIGHT.

    # dir = os.listdir(vid_path)
    # # frames_wanted = 20
    # new = np.empty((20, IM_WIDTH, IM_HEIGHT, 3))
    # for i in tqdm(np.arange(len(dir))):
    #     video = vread(vid_path + '/' + dir[i])
    #     for j in np.arange(len(video)):
    #         new[j] = cv2.resize(video[j], (IM_WIDTH, IM_HEIGHT), interpolation=cv2.INTER_CUBIC)
    #     vwrite(vid_path + '2/' + dir[i], new)

    # Resetting the video path to the newly created folder with new image resolution.
    # If previous code snippet was not used and image resolution is still 200x200, use previous vid_path.
    vid_path = r'C:\Users\dekel\OneDrive\Final Project\Mor_Deep\Masks_Final2'

    # Loading the video folder to memory.
    dir = os.listdir(vid_path)
    X = []
    for i in tqdm(np.arange(len(dir))):
        video = vread(vid_path + '/' + dir[i])
        # Videos are saved with 3 channels, and here we need only the first 2.
        video = video[:, :, :, :2]
        X.append(video)
    X = np.asarray(X)

    # First train/test split.
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.4, stratify=y,
                                                        shuffle=True, random_state=42)

    # Splitting training data to training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, stratify=y_train_temp,
                                                        shuffle=True, random_state=42)

    # One-Hot encoding the training and validation data.
    y_train_data = [to_categorical(y_train[:, 0], 4), to_categorical(y_train[:, 1], 2),
                    to_categorical(y_train[:, 2], 2)]
    y_val_data = [to_categorical(y_val[:, 0], 4), to_categorical(y_val[:, 1], 2),
                  to_categorical(y_val[:, 2], 2)]

    # Declaring the number of frames in each video. Can change by changing the Split_video.py script.
    N = 20

    # Declaring the model specs.
    model = CovidMultiOutputModel(input_size=(N, IM_WIDTH, IM_HEIGHT, 2),  # Does not include batch size.
                                  hidden_blocks=3,  # Number of hidden blocks.
                                  filters=[48, 48, 48],  # Conv filter sizes.
                                  kernel_sizes=[(2, 3, 3), (2, 3, 3), (1, 3, 3)],  # Conv kernel sizes.
                                  paddings=["valid"]*3,  # Type of conv padding.
                                  activations=["relu"]*3,  # Type of activation between each conv layer.
                                  batchnorms=[False]*3,  # True if we want batch normalization layers, False if not.
                                  pool_sizes=[(3, 3, 3), (2, 3, 3), (2, 3, 3)],  # MaxPooling kernel sizes.
                                  dropouts=[0.15, 0.15, 0.15],  # Dropout values.
                                  # Params for COVID-19 severity grade branch.
                                  covid_params={'dense': 128, 'activation': "relu", 'batchnorm': False,
                                                'dropout': 0.2, 'final_activation': "softmax"},
                                  # Params for pleural line regularity branch.
                                  pleural_params={'dense': 128, 'activation': "relu", 'batchnorm': False,
                                                  'dropout': 0.2, 'final_activation': "sigmoid"},
                                  # Params for consolidation branch.
                                  consolidation_params={'dense': 128, 'activation': "relu", 'batchnorm': False,
                                                  'dropout': 0.2, 'final_activation': "sigmoid"}).assemble_full_model()

    # Model Summary.
    print(model.summary())

    # Params for model training.
    init_lr = 1e-3
    epochs = 50
    decay = init_lr / epochs
    opt = Adam(learning_rate=init_lr, decay=decay)

    # Compiling the model.
    model.compile(optimizer=opt,
                  # Loss function declaration.
                  loss={
                      'covid_severity_output': 'categorical_crossentropy',
                      'pleural_regular_output': 'binary_crossentropy',
                      'consolidation_output': 'binary_crossentropy'},
                  # Relative value of the branch loss to the total loss.
                  # The model tries to minimize the total loss as a combination of the branch losses.
                  loss_weights={
                      'covid_severity_output': 2,
                      'pleural_regular_output': 1,
                      'consolidation_output': 1},
                  # Scoring metrics we want to record in the history.history variable
                  metrics={
                      'covid_severity_output': ['categorical_accuracy', 'Precision', 'Recall', 'AUC', 'mae'],
                      'pleural_regular_output': ['binary_accuracy', 'Precision', 'Recall', 'AUC', 'mae'],
                      'consolidation_output': ['binary_accuracy', 'Precision', 'Recall', 'AUC', 'mae']})

    # Saving the model
    tf.keras.models.save_model(model=model, filepath='model.h5', save_format='h5')

    # Batch size. GeForce RTX 2060 could only handle batch_size = 5.
    batch_size = 5

    callbacks = [
        ModelCheckpoint("./model_checkpoint", monitor='val_loss')
    ]

    # Training process.
    history = model.fit(x=X_train, y=y_train_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val_data))
    # Plot all of the metrics fro train/validation sets for each epoch (using Plotly).
    calc_params_train_val(history)
    # Plot the total loss for each epoch (using Plotly).
    calc_loss_train_val(history)

    # One-Hot encoding test data.
    y_test_data = [to_categorical(y_test[:, 0], 4), to_categorical(y_test[:, 1], 2), to_categorical(y_test[:, 2], 2)]
    # Evaluating the model using the test data.
    results = model.evaluate(X_test, y_test_data, batch_size=batch_size)
    print(results)

    # True labels.
    covid_severity_true, pleural_regular_true, consolidation_true = y_test[:, 0], y_test[:, 1], y_test[:, 2]

    # Predicted labels.
    covid_severity_pred, pleural_regular_pred, consolidation_pred = [], [], []
    for i in np.arange(len(X_test)):
        pred_temp = model.predict(X_test[i:(i + 1)])

        covid_severity_pred.append(np.argmax(pred_temp[0]))
        pleural_regular_pred.append(np.argmax(pred_temp[1]))
        consolidation_pred.append(np.argmax(pred_temp[2]))
    covid_severity_pred = np.asarray(covid_severity_pred)
    pleural_regular_pred = np.asarray(pleural_regular_pred)
    consolidation_pred = np.asarray(consolidation_pred)

    # CNN output confusion matrix
    Get_CNN_Confusion_Matrix(covid_severity_pred, covid_severity_true, param_name='covid_severity_grade')
    Get_CNN_Confusion_Matrix(pleural_regular_pred, pleural_regular_true, param_name='pleural_line_regular')
    Get_CNN_Confusion_Matrix(consolidation_pred, consolidation_true, param_name='consolidation')

    # Training a RFC to further improve COVID-19 severity grade results.

    # Creating the feature vector by concatenating CNN output of test data
    pred_RFC = model.predict(X_test, batch_size=batch_size)
    feat_RFC = np.concatenate((pred_RFC[0], pred_RFC[1], pred_RFC[2]), axis=1)

    # Splitting the test data into train/test.
    X_train_RFC, X_test_RFC, y_train_RFC, y_test_RFC = train_test_split(feat_RFC, covid_severity_true,
                                                                  test_size=0.4, stratify=covid_severity_true,
                                                                  shuffle=True, random_state=42)

    # Function that creates the RFC model, trains it, and returns a confusion matrix & other scoring metrics.
    rfc = Random_forest_classifier(X_train_RFC, X_test_RFC, y_train_RFC, y_test_RFC)

    # Saving the RFC model
    filename = 'RFC_model.sav'
    pickle.dump(rfc, open(filename, 'wb'))
