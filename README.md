# landsat-classification

<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/greenland_oli_2021189.jpeg?raw=true" width ="1000" height=500>

## Background
NASA has launched 8 out of 9 Landsat satellites to provide continous imagery of Earth.  The imagery is used for several applications, including:

<li>Agriculture</li>
<li>Forest Management</li>
<li>Disaster Assessment</li>
<li>Forest Management.</li>

## Data
The data are available on the <a href="https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29">UCI Machine Learning Repository</a>.  The data files are also contained in the <a href="https://github.com/jstodd867/landsat-classification/tree/main/data">data directory</a> of this repository.

The dataset contains 4435 training samples and 2000 test samples.  Each sample is a 36 element feature vector that contains a 3x3x4 pixel neighborhood across 4 channels, as illustrated in the figure below.

<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/samples.png" width ="350" height=250>

## Objective
The goal is to correctly predict which class each center pixel belongs to, from the following class labels:

<li>Red soil</li>
<li>Cotton crop</li>
<li>Grey soil</li>
<li>Damp grey soil</li>
<li>Soil with vegetation</li>
<li>Very damp grey soil</li>

## Code Organization

The code is in the <a href="https://github.com/jstodd867/landsat-classification/tree/main/src">src directory</a> and is organized as follows:

<li><a href="https://github.com/jstodd867/landsat-classification/blob/main/src/data.py">data.py</a>:  helper functions to load and pre-process data</li>
<li><a href="https://github.com/jstodd867/landsat-classification/blob/main/src/models.py">models.py</a>:  class and helper functions for various models</li>
<li><a href="https://github.com/jstodd867/landsat-classification/blob/main/src/plots.py">plots.py</a>:  helper functions for common plots</li>
<li><a href="https://github.com/jstodd867/landsat-classification/blob/main/src/run_models.py">run_models.py</a>:  main script to run to re-produce results</li>

There is also a jupyter notebook in the root directory that contains the same code as in the run_models.py script.

## Exploratory Data Analysis

To get a sense of the class balance, the number of instances of each class was plotted, shown below.
<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/train_class_count.png">

The classes are fairly balanced, though 3 of them have roughly twice the amount of instances as the other 3.  The test data have a very similar distribution among the classes.

Next, histograms of intensity values for the center pixels of each channel were plotted to get a sense of the distribution of the values.  From the plot below, it can be seen that each channel has some variation in the shapes and center values of the distributions.

<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/ctr_pix_histogram.png">

## Classification Models

### Baseline Model
To provide a baseline for performance, a simple model was created that predicts the class of each center pixel as the one that has the smallest Euclidean distance to the mean class feature vector.  This model resulted in a decent baseline accuracy of 76.85%.

### Random Forest Classifier
A random forest model was built and tuned to the dataset.  A gridsearch was performed to find the optimal hyperparameters.  The following parameters were included in the gridsearch:

<li>n_estimators</li>
<li>max_depth</li>
<li>min_samples_split</li>
<li>min_samples_leaf</li>

### Deep Neural Network
A deep neural network was built and tuned to the dataset.  A gridsearch was performed to find the optimal hyperparameters.  The following parameters were included in the gridsearch:

<li>learning rate</li>
<li>number of hidden layers</li>
<li>number of units in hidden layers</li>
<li>activation functions of hidden layers</li>
<li>batch size</li>
<li>number of epochs</li>

### Results - Model Performance Summary
The accuracy of each of the models are listed at the top of each subplot, below.
<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/confusion_matrices.png">
The Random Forest classifier and the deep neural network achieved significantly improved accuracy (91%) beyond that of the baseline model.  However, both models had a notable amount of misclassification errors, mostly between classes 4 (damp grey soil) and 7 (very damp grey soil) and classes 3 (grey soil) and 4 (damp grey soil).  Looking at the labels of these classes, the likely reason for the misclassifications is evident:  all 3 classes are variants of grey soil with different levels of dampness.  The remaining classes are actually different types of material (e.g., red soil).

## Conclusions
Both the Random Forest and Neural Network classifiers achieved approximately 91% accuracy on the test set.  The largest misclassification errors are understandably in categories that are very similar (grey soil, damp grey soil, and very damp grey soil).  Training on more data and/or ensembling the 2 different classifiers may yield improvements.
