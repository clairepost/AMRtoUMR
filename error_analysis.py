
#using stratified k-fold cross validation




def run_buckets(train, test):
    # combine training and test data
    # using a seed- split into 5 different 80-20 splits?
    # for each bucket: train model and test
        # report accuracy
        #report the final layer of the weights - that's how much we should be able to trust the model
        # report confusion matrix
    pass

from sklearn.model_selection import StratifiedKFold
import numpy as np


def get_indices(x):
    num_splits = 5
    seed = 0
    kf = StratifiedKFold(n_splits=num_splits, shuffle = True, random_state=seed)
    kf.get_n_splits(x,x)
    return enumerate(kf.split(x,x))

