# catML
A Machine Learning project to predict star ratings from Yelp reviews.

NOTE: The training dataset is too large to be uploaded onto Github. Download data_train.json [here](https://drive.google.com/open?id=1rplZW9D-kkY879lI50RbJ6lzId0Am9ms).

There will be a file in the data folder (after I finish running data_preprocessing.py) called __trainingData.csv__ with the following format:

Labels | at | app | f3 | f4 
--- | --- | --- | --- |---
2 | 0 | 0 | 2 | 1 
1 | 0 | 0 | 1 | 0 

Each instance will be represented in this feature vector format we are familiar with, so make sure that your model's code will accept this input for data instances.
