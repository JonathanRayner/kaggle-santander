import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class FrequencyFeatures:
    """
    Tools to create and apply a lookup table that maps dataframe entries to the frequency of those entries in the training data. Frequencies are normalized so that maximum frequency = 1.

    Attributes:
        lookup (pandas dataframe): Dataframe were indices are entries, values are corresponding frequencies.
        decimals (int): Number of decimal places of desired precision when rounding and binning for 'fit' and 'transform'
    """

    def __init__(self):
        self.lookup = None
        self.decimals = None

    def bin_as_int(self, dataframe, decimals):
        """
        Rounds dataframe entries to desired level of precision (as a form of binning), multiplies by appropriate power of 10 and converts to ints to avoid problems with floating point errors later.

        Args:
            dataframe (pandas dataframe): Dataframe to be binned as int.
            decimals (int): Number of decimal places of desired precision when rounding.

        Returns:
            pandas dataframe of ints
        """
        # multiply 'dataframe' by 10 to the power 'decimals'
        binned_dataframe = dataframe.copy()
        binned_dataframe = binned_dataframe*(10**decimals)

        # round away all decimal places and convert to int
        binned_dataframe = binned_dataframe.round(decimals=0).astype(int)

        return binned_dataframe

    def fit(self, train_dataframe, decimals):
        """
        Creates frequency lookup dataframe from training data and stores it as a class variable, so that it can be used to transform any dataframe. Frequencies are normalized so that max frequency = 1.

        Args:
            train_dataframe (pandas dataframe): Dataframe to use to create lookup table.
            decimals (int): Number of decimal places of desired precision when rounding.

        Returns:
            None
        """
        # store decimals as class variable to be used later in 'transform'
        self.decimals = decimals

        # bin train_dataframe into bins of int type
        binned_dataframe = self.bin_as_int(dataframe=train_dataframe, decimals=self.decimals)

        # flatten 'binned_dataframe' into a single series, then count values and sort by values
        lookup = binned_dataframe.melt()['value'].value_counts().sort_index()

        # normalize by maximum
        lookup = lookup/lookup.max()

        # cut off very small values by rounding, then taking nonzero entries. Reduces size of lookup dataframe
        # choose convention of 4 decimal places, because this is level of precision of raw data in this problem
        lookup = lookup.round(decimals=4)
        lookup = lookup[lookup != 0]

        # linearly interpolate any missing rows, so that we can handle any unseen data
        # create new index with no missing values from first index to last
        filled_index = pd.RangeIndex(start=lookup.index[0], stop=lookup.index[-1] + 1)

        # fill in all missing indices with NaN
        lookup = lookup.reindex(index=filled_index)

        # replace NaNs with linear interpolated values
        lookup = lookup.interpolate()

        # store as a class variable
        self.lookup = lookup

        return None

    def transform(self, dataframe):
        """
        Creates a dataframe from 'dataframe' where entries have been mapped to frequencies according to 'self.lookup'.

        Args:
            dataframe (pandas dataframe): Dataframe to be transformed.

        Returns:
            Dataframe transformed according to 'self.lookup'.
        """
        dataframe = self.bin_as_int(dataframe=dataframe, decimals=self.decimals)

        # transform lol according to lookup, then replace any missing values with zero (these will be on the tails of the lookup distribution)
        # applying transformation column-wise using .apply() and .map() is a performance improvement vs. other methods such as using .replace()
        dataframe = dataframe.apply(lambda x: x.map(self.lookup)).fillna(value=0)

        return dataframe

def change_class_balance(train_dataframe, train_labels, oversample_rate, majority_class_ratio):
    """
    Rebalance classes in the training data by oversampling classes with label 1 (duplicating, with replacement) and undersampling classes with label 0 (without replacement).

    Args:
        train_dataframe (pandas dataframe): Training data.
        train_labels (pandas dataframe): Training data labels.
        oversample_rate (float): Percentage of class 1 examples to duplicate.
        majority_class_ratio (float): Multiplies class 1 size to give class 0 size.

    Returns:
        Dataframe with balanced classes, Dataframe with corresponding labels
    """
    train_augmented = train_dataframe.copy()
    train_augmented['target'] = train_labels.copy()

    pos_examples = train_augmented.loc[train_augmented['target']==1]
    neg_examples = train_augmented.loc[train_augmented['target']==0]

    # oversample positive examples with replacement, according to 'oversample_rate'
    num_pos = len(pos_examples.index)
    num_additional_pos_examples = int(oversample_rate*num_pos)

    # duplicate examples to be added
    duplicate_pos_examples = pos_examples.sample(n=num_additional_pos_examples, replace=True)

    # combine duplicate positive examples with original positive examples
    pos_examples = pd.concat([pos_examples, duplicate_pos_examples])

    # choose class balance by undersampling the same number of negative examples as augmented positive examples, without replacement
    num_neg = int(majority_class_ratio*len(pos_examples.index))
    neg_examples = neg_examples.sample(n=num_neg, replace=False)

    # combine positive and negative examples, shuffle, reindex, split off targets
    train_augmented = pd.concat([pos_examples, neg_examples])
    train_augmented = train_augmented.sample(frac=1, replace=False).reset_index(drop=True)

    train_labels_augmented = train_augmented.pop('target')

    return train_augmented, train_labels_augmented

def pair_features(array):
    """
    Pairs each original feature value with the frequency feature value made from it.

    Args:
        array (ndarray): Numpy array of shape (number of data points, 400)

    Returns:
        Reshaped numpy array of shap (number of data points, 200, 2), where the first column of the old array is paired with the 200th column of the old array, 2nd column paired with 201st column, etc.
    """
    print("Initial shape ", array.shape)

    # first 200 features form row 1, last 200 features form row 2, for each data point
    array = np.reshape(array, (-1, 2, 200))

    # transpose only axis 2 and 3 so that each data point is transposed
    array = np.transpose(array, (0,2,1))

    print("Final shape ", array.shape)

    return array

def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.figure(figsize=(6,4))

    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color='b', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color='g', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

    plt.legend()
    plt.savefig('metrics.png')
    plt.close()
