import models
import plots
import data
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

if __name__ == '__main__':
    # Load the training and test data
    X_train, X_test, y_train, y_test = data.load_data('data')

    # Plot occurrences of each class in train and test data
    fig, axs = plt.subplots(2,1,figsize=(7,10))
    plots.class_bar_plot(axs[0], y_train, 'Occurrences of Each Class in Training Set', 'Class', 'Number of Occurrences')
    plots.class_bar_plot(axs[1], y_test, 'Occurrences of Each Class in Training Set', 'Class', 'Number of Occurrences')
    fig.tight_layout()
    plt.show()

    # Plot a histogram of intensity values for the center pixel of each channel
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(X_train[:,19], color='green', alpha=0.6, label='Channel 4')
    ax.hist(X_train[:,16], alpha = 0.6, label='Channel 1')
    ax.hist(X_train[:,17], color='orange', alpha=0.6, label='Channel 2')
    ax.hist(X_train[:,18], color='gray', alpha=0.6, label='Channel 3')

    ax.set_title('Histogram of Intensity Values of Center Pixel By Channel')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Counts')

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    plt.show()

    # Fit and evaluate baseline model
    base_model = models.Baseline() 
    base_model.fit(X_train[:,16:20], y_train)  #Fit model using only center pixels

    y_test_predict = base_model.predict(X_test[:,16:20])
    print(f'Baseline Model Accuracy: {accuracy_score(y_test,y_test_predict)}')

    # Scale X data and one-hot encode y data for use in neural network
    X_trn_scaled, X_test_scaled, y_trn_1hot, y_test_1hot = data.prep_data_nn(X_train, X_test, y_train, y_test)

    # Set parameters for neural network architecture and training
    opt = keras.optimizers.Adam(learning_rate=0.0003)
    hidden_units = 100
    n_classes = 6
    n_hidden = 1
    es = keras.callbacks.EarlyStopping(monitor='loss', patience=100) # Set early stopping

    # Fit and evaluate neural network
    nn_model = models.create_model(X_trn_scaled, n_classes, n_hidden, opt, hidden_units, activ='exponential')
    history = nn_model.fit(X_trn_scaled, y_trn_1hot, epochs=300, batch_size=20, verbose=1, validation_split=0, callbacks=es)
    
    nn_model.evaluate(X_test_scaled, y_test_1hot)  # Calculate accuracy on unseen test data

    # Perform gridsearch for optimal neural network parameters
    opt_nn_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=models.create_model, X=X_trn_scaled, n_classes=6, opt=opt)
    # Set grid search parameters
    nn_param_grid = dict(epochs=[700], batch_size=[20], n_hidden=[1,2], hidden_units=[20,35, 50], activ=['sigmoid','exponential'])
    # Create grid and execute search
    grid = GridSearchCV(estimator=opt_nn_model, param_grid=nn_param_grid, n_jobs=-1, cv=[(slice(None), slice(None))])
    grid_result = grid.fit(X_trn_scaled, y_trn_1hot)
    # Print results
    print(f'Best Neural Network Parameters from Gridsearch: {grid_result.best_params_}')  #print parameters of best neural network
    grid_result.best_estimator_.model.evaluate(X_test_scaled, y_test_1hot)  # Calculate performance on unseen test data
    # Predict y values for test set
    grid_result.best_estimator_.classes_= np.array([1,2,3,4,5,7]) #reset class labels
    yhat_nn= grid_result.best_estimator_.predict(X_test_scaled)

    # Random Forest Classifer
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0, class_weight='balanced')
    rf_clf.fit(X_train, y_train)       #Train the model
    y_trn_rf = rf_clf.predict(X_train)
    print(f'Random Forest Train Accuracy: {accuracy_score(y_train, y_trn_rf)}')
    y_rf = rf_clf.predict(X_test)      #Predict values for test set
    print(f'Random Forest Test Accuracy: {accuracy_score(y_test, y_rf)}')

    # Random Forest Gridsearch
    # Construct grid of parameters for search
    rf_param_grid = dict(n_estimators=[10, 50, 100], min_samples_split=np.arange(2,10,2),
    min_samples_leaf=np.arange(1,10,2), max_depth = np.arange(10,30,10),
    class_weight=['balanced'])

    # Perform search
    grid = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, n_jobs=-1, cv=5,scoring='f1_weighted')
    rf_grid_result = grid.fit(X_train, y_train)

    print(f'Best Random Forest Parameters: {rf_grid_result.best_params_}\n')
    print(f'Average Validation Accuracy: {rf_grid_result.best_score_}')

    y_rf = rf_grid_result.best_estimator_.predict(X_test)
    print(f'Random Forest Test Accuracy Score: {accuracy_score(y_test, y_rf)}')

    # Compare model performance
    # Compare confusion matrices
    fig, axs = plt.subplots(1,3,figsize=(20,5))
    # Plot confusion matrix for each classifier
    plots.plot_cm(y_test, y_test_predict, [1,2,3,4,5,7], axs[0])
    plots.plot_cm(y_test, y_rf, rf_grid_result.best_estimator_.classes_, axs[1])
    plots.plot_cm(y_test, yhat_nn, grid_result.best_estimator_.classes_, axs[2])
    # Label subplots
    axs[0].set_title(f'Baseline\n(Accuracy = {np.round(accuracy_score(y_test, y_test_predict),2)})')
    axs[1].set_title(f'Random Forest\n(Accuracy = {np.round(accuracy_score(y_test, y_rf),2)})')
    axs[2].set_title(f'Neural Network\n(Accuracy = {np.round(accuracy_score(y_test, yhat_nn),2)})')
    plt.show()
