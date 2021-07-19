import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall


class Baseline():
    
    def __init__(self):
        self.classes_ = None
        self.classes_inverse_ = None
        self.means_ = None
        self.mean_dict = None
        self.label_dict = None
    
    def predict(self, X):
        predicts = []
        self.label_dict = {num:label for num,label in zip(np.arange(len(self.classes_)), self.classes_)}

        # Calculate and store closest class spectral vector for each data sample
        for sample in X:
            mean_index = np.argmin([np.sqrt(sum((sample - mean_vector)**2)) for mean_vector in self.mean_dict.values()])
            predicts.append(self.label_dict[mean_index])

        return np.array(predicts)
    
    def fit(self, X_train, y_train):
        self.classes_, self.classes_inverse_ = np.unique(y_train, return_inverse=True)
        self.means_ = [np.mean(X_train[self.classes_inverse_==idx,:], axis=0) for idx,label in enumerate(self.classes_)]
        self.mean_dict = {label: self.means_[i] for i,label in enumerate(self.classes_)}

def create_model(X, n_classes, n_hidden=3, opt='Adam', hidden_units=100, drop_out=0, activ='sigmoid'):
    '''Create a deep neural network model
    
    Parameters
    ----------
    X: array
        X data that will be used to train the network
    n_classes: int
        Number of classes to predict
    n_hidden: int
        Number of hidden layers
    opt: keras optimizer object
        Optimizer to use with neural network
    hidden_units: int
        Number of hidden units to use in the hidden layers
    drop_out: float
        dropout rate
    activ: str
        Activation function for hidden layer units

    Returns
    -------
    model: keras model object
        A compiled model
    '''
    np.random.seed(42)

    n_samples, n_feats = X.shape # Get shape of input data to construct network

    model = Sequential() 

    hidden_layer = Dense(units=hidden_units,
                    input_dim=n_feats,
                    kernel_initializer='constant',
                    activation='softsign')

    # Add first hidden layer, if required
    if n_hidden>0:
        model.add(Dense(units=hidden_units,
                    input_dim=n_feats,
                    kernel_initializer='uniform',
                    activation=activ))
        #model.add(Dropout(drop_out))
    else:
        hidden_units=n_feats

    # Define output layer
    outputlayer = Dense(units=n_classes,
                    input_dim=hidden_units,
                    kernel_initializer='uniform',
                    activation='softmax')

    # Add hidden layers to model, if required
    if n_hidden>1:
        for _ in np.arange(1,n_hidden):
            model.add(Dense(units=hidden_units,
                    kernel_initializer='uniform',
                    activation=activ))
            #model.add(Dropout(drop_out))

    # Add output layer 
    model.add(outputlayer)

    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, metrics=["accuracy"])
    return model