import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_data(data_path):
    ''' Load train and test data from the files given by the data_path parameter'''
    train, test = np.loadtxt(os.path.join(data_path, 'sat.trn')), np.loadtxt(os.path.join(data_path, 'sat.tst'))
    # Separate labels into different vectors
    y_train, y_test = train[:,-1].astype('int'), test[:,-1].astype('int')
    # Separate 3x3x4 pixel neighborhood samples into train and test feature vectors
    X_train, X_test = train[:,:-1], test[:,:-1]
    return X_train, X_test, y_train, y_test

def prep_data_nn(X_train, X_test, y_train, y_test):
    # One hot encode the target data for use with the neural network 
    enc = OneHotEncoder()
    enc.fit(y_train.reshape(-1,1))
    y_trn_1hot, y_test_1hot = enc.transform(y_train.reshape(-1,1)).toarray(), enc.transform(y_test.reshape(-1,1)).toarray()
    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_trn_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    return X_trn_scaled, X_test_scaled, y_trn_1hot, y_test_1hot

if __name__ == '__main__':
    # Load training and test data from files
    X_train, X_test, y_train, y_test = load_data('data')

    # Scale X data and one-hot encode y data for use in neural network
    X_trn_scaled, X_test_scaled, y_trn_1hot, y_test_1hot = prep_data_nn(X_train, X_test, y_train, y_test)

    