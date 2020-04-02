
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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

# create dataset for lightgbm
lgb_train = lgb.Dataset(train, train_labels)
lgb_eval = lgb.Dataset(validation,
                       validation_labels,
                       reference=lgb_train)

random_state = 42

learning_rate_schedule = [0.1]*1000 + [0.01]*2000 + [0.002]*10000 + [0.001]*(20000-1000-2000-10000)

params = {
    # default num_trees=100
    'num_trees': 20000,
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 4,
    'learning_rate': 0.002,
    'boost_from_average': 'false',
    # Percentage of features to be used for each tree
    'feature_fraction': 0.10,
    'min_data_in_leaf': 80,
    # Percentage of data to be sampled for each tree
    'bagging_fraction': 0.4,
    # Perform bagging at every k-th tree (bagging_freq must be non-zero for bagging_fraction to be used)
    'bagging_freq': 5,
    # Documentation recommends using number of available cores, not number of available threads
    'num_threads': 7,
    'bagging_seed' : random_state,
    'seed': random_state
}

print('Starting training...')

# train
original_model = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                           early_stopping_rounds=2000,
                           callbacks=[lgb.reset_parameter(learning_rate=learning_rate_schedule)]
)

print('Done Training.')
