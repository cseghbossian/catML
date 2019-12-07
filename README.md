# catML
## Group Info
Group 2
- Angela Agabin \<aagabin\> ID: 1590090
- Celine Seghbossian \<cseghbos\> ID: 1595557
- Kaleen Shrestha \<kashrest\> ID: 1615749
  
## Overview
A Machine Learning project to predict star ratings from Yelp reviews. This program uses five different models to conduct its multiclass classifications:
* Logistic Regression
* Perceptron
* Support Vector Machine
* Nearest Centroid 
* Voting Ensemble of the above four

## List of Files in Submission
- Report.pdf - a report explaining the project
- AgabinSeghbossianShrestha_predictions.csv - the final predictions for the provided test 
- AgabinSeghbossianShrestha_code.zip - the zip file of all source code
  - pickled_feature_vectorizer - the TFID vectorizer used to extract a set of features from text instances
  - pickled_svm - pickled SVM model
  - pickled_perceptron - pickled perceptron model
  - pickled_nc - pickled nearest centroid model
  - pickled_logistic_regression - pickled logistic regression model
  - feature_extractions_BOW.ipynb - python notebook with code related to feature extraction
  - perceptron_nearest_centroid.ipynb - python notebook with code related to perceptron and nearest centroid classifiers
  - SVM.ipynb - python notebook with code related to SVM classifier
  - logistic_regression.ipynb - python notebook with code related to logistic regression model
  - predictor_client.py - python file to be used for testing models
  
## Instructions for Testing a Dataset
1. Unzip AgabinSeghbossianShrestha_code.zip 
2. Run predictor_client.py with the filepath to the testing dataset as the only argument to the program
3. The program will use a voting ensemble of four different models to output its predictions to predictions_group2.csv
NOTE: Do not run any of the ipynb files. Many of them have long runtimes and rely on certain data files to be inside the working directory.
