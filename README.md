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

The dataset contains 4435 training samples and 2000 test samples.  Each sample is a 36 element feature vector that contains a 3x3x4 pixel neighborhood across 4 channels.

## Objective
The goal is to correctly predict which class each center pixel belongs to, from the following class labels:

<li>Red soil</li>
<li>Cotton crop</li>
<li>Grey soil</li>
<li>Damp grey soil</li>
<li>Soil with vegetation</li>
<li>Very damp grey soil</li>

## Exploratory Data Analysis

To get a sense of the class balance, the number of instances of each class was plotted, shown below.
<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/train_class_count.png">

The classes are fairly balanced, though 3 of them have roughly twice the amount of instances as the other 3.  The test data have a very similar distribution among the classes.

Next, histograms of intensity values for the center pixels of each channel were plotted to get a sense of the distribution of the values.  From the plot below, it can be seen that each channel has some variation in the shapes and center values of the distributions.

<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/ctr_pix_histogram.png">

## Classification Models

### Baseline Model
To provide a baseline for performance, a simple model was created that predicts the class of each center pixel as the one that has the smallest Euclidean distance to the mean class feature vector.

### Random Forest Classifier

### Deep Neural Network

### Model Performance Summary
<img src="https://github.com/jstodd867/landsat-classification/blob/main/imgs/confusion_matrices.png">

## Conclusions

