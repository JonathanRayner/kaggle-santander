import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
# custom tools
from tools import FrequencyFeatures, FrequencyFeaturesPerColumn, pair_features, plot_metrics

# Import and split the data
train = pd.read_csv('train.csv')

# The ID_code column contains no information, so we remove it
train.drop('ID_code', axis=1, inplace=True)

train, validation = train_test_split(train, test_size=0.2)

train_labels = train.pop('target')
validation_labels = validation.pop('target')

# Preprocessing
# make a copy of train and and validation, scaled by z-score
train_scaled = train.copy()
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_scaled)
train_scaled = pd.DataFrame(train_scaled).reset_index(drop=True)

validation_scaled = validation.copy()
validation_scaled = scaler.transform(validation_scaled)
validation_scaled = pd.DataFrame(validation_scaled).reset_index(drop=True)

# Create frequency features
# comment and uncomment as needed to use frequency distribution of whole of training data, or separate distribution per feature
frequency_features = FrequencyFeatures()
#frequency_features_per_column = FrequencyFeaturesPerColumn()

# number of decimal places we'll round everything to, implementing binning
decimals = 1
#decimals_per_column = 4

frequency_features.fit(train_dataframe=train, decimals=decimals)
#frequency_features_per_column.fit(train_dataframe=train, decimals=decimals_per_column)

train_new_features = frequency_features.transform(dataframe=train).reset_index(drop=True)
validation_new_features = frequency_features.transform(dataframe=validation).reset_index(drop=True)

#train_new_features_pc = frequency_features_per_column.transform(dataframe=train).reset_index(drop=True)
#validation_new_features_pc = frequency_features_per_column.transform(dataframe=validation).reset_index(drop=True)

# combine scaled features and new features (we reset the index of everything earlier)
train = pd.concat([train_scaled, train_new_features], axis=1)
validation = pd.concat([validation_scaled, validation_new_features], axis=1)

#train = pd.concat([train_scaled, train_new_features_pc], axis=1)
#validation = pd.concat([validation_scaled, validation_new_features_pc], axis=1)

# tensorflow needs numpy arrays
train = np.array(train)
validation = np.array(validation)

train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)

# reshape so that new features are paired with corresponding original features
train = pair_features(train)
validation = pair_features(validation)

# Conv net needs (amount of data, height, width, channels) format. Here channels = 1
def conv_reshape(data):
    data_reshape = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return data_reshape

train = conv_reshape(train)
validation = conv_reshape(validation)

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
BATCH_SIZE = 256
EPOCHS = 40
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

def optimizer():
    return tf.keras.optimizers.Adam(lr=0.001, clipnorm=1.0)

# whole dataset decimals = 1 seems best, 2, 3 also good. decimals = 1 on whole dataset seems best out of all per column and whole dataset options
# per column, decimals = 3 auc 0.8996 architecture. decimals 3 seems to perform slightly better than decimals = 4
# staircase scheduler
learning_rates = [0.001]*20 + [0.0005]*10 + [0.0001]*10
lr_schedule_staircase = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rates[epoch], verbose=1)

def make_model(train_features, metrics=METRICS):
    model = keras.Sequential([
        # Note input shape (200, 2, 1) when we pair features with frequency features
        keras.layers.Conv2D(256, (1,2), kernel_regularizer=keras.regularizers.l2(0.0001), activation='relu',
                           input_shape=(train_features.shape[1], train_features.shape[2],1)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        # No Pooling layers needed, because already as small as can be
        keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001),
                     activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001),
                     activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001),
                     activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.0001),
                     activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer=optimizer(),
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
model.summary()

# train
model_history = model.fit(
    train,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # callbacks=[early_stopping],
    callbacks=[lr_schedule_staircase],
    validation_data=(validation, validation_labels))

# save plot of metrics
plot_metrics(model_history)
