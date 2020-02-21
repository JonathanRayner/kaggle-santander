import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# import preprocessing and plotting helper functions
import nn_functions as nnf

# Import and split the data
train = pd.read_csv('train.csv')

# The ID_code column contains no information, so we remove it
train.drop('ID_code', axis=1, inplace=True)

train, validation = train_test_split(train, test_size=0.2)

train_labels = train.pop('target')
validation_labels = validation.pop('target')

# Preprocessing
# make a copy of trainand and validation, scaled to [0,1]
train_scaled = train.copy()
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_scaled)
train_scaled = pd.DataFrame(train_scaled).reset_index(drop=True)

validation_scaled = validation.copy()
validation_scaled = scaler.transform(validation_scaled)
validation_scaled = pd.DataFrame(validation_scaled).reset_index(drop=True)

# Create frequency features
frequency_features = nnf.FrequencyFeatures()

# number of decimal places we'll round everything to
decimals = 1

frequency_features.fit(train_dataframe=train, decimals=decimals)

train_new_features = frequency_features.transform(dataframe=train).reset_index(drop=True)
validation_new_features = frequency_features.transform(dataframe=validation).reset_index(drop=True)

# combine scaled features and new features (we reset the index of everything earlier)
train = pd.concat([train_scaled, train_new_features], axis=1)
validation = pd.concat([validation_scaled, validation_new_features], axis=1)

# uncomment to use class balancing
# majority_class_ratio = 7.0
# oversample_rate = 0.25

# train, train_labels = nnf.change_class_balance(train_dataframe=train,
#                                                    train_labels=train_labels,
#                                                    oversample_rate=oversample_rate,
#                                                    majority_class_ratio=majority_class_ratio)

# tensorflow needs numpy arrays
train = np.array(train)
validation = np.array(validation)

train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

# reshape so that new features are paired with corresponding original features
train = nnf.pair_features(train)
validation = nnf.pair_features(validation)

# clear keras session so that we can rerun without errors
tf.keras.backend.clear_session()

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

N_TRAIN = train.shape[0]
BATCH_SIZE = 2048
EPOCHS = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

# take some ideas from https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

# learning rate decay
# initial learning rate
initial_rate = 0.005

# 'decay_factor' = x means learning rate decays to 1/2 of the 'initial_rate' after x epochs, 1/3 after 2x epochs, etc.
decay_factor = 20

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_rate,
    decay_steps=STEPS_PER_EPOCH*decay_factor,
    decay_rate=1,
    staircase=False)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

# model architecture
def make_model(train_features, metrics=METRICS):
    model = keras.Sequential([
        # Note input shape (200, 2) when we pair features with frequency features
        keras.layers.Dense(64,
                           kernel_regularizer=keras.regularizers.l2(0.0001),
                           activation='elu',
                           input_shape=(train_features.shape[1], train_features.shape[2])),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.0001),
                     activation='elu'),
        keras.layers.Dropout(0.2),
        # keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.4),
        # keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.2),
        # keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.0001),
        #              activation='elu'),
        # keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer=get_optimizer(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)
    return model

# uncomment here and in model.fit to use early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_auc',
#     verbose=1,
#     patience=10,
#     mode='max',
#     restore_best_weights=True)

# initialize the model
model = make_model(train)

# train
# Features and labels input as numpy arrays
model_history = model.fit(
    train,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # callbacks=[early_stopping],
    validation_data=(validation, validation_labels))

# Display metrics plot
nnf.plot_metrics(model_history)
