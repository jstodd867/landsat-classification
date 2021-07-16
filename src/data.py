import os
import numpy as np

def load_data(data_path):
    ''' Load train and test data from the files given by the data_path parameter'''
    train, test = np.loadtxt(os.path.join(data_path, 'sat.trn')), np.loadtxt(os.path.join(data_path, 'sat.tst'))
    # Separate labels into different vectors
    y_train, y_test = train[:,-1].astype('int'), test[:,-1].astype('int')
    # Separate 3x3x4 pixel neighborhood samples into train and test feature vectors
    X_train, X_test = train[:,:-1], test[:,:-1]
    return X_train, X_test, y_train, y_test
