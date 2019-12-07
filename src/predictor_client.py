#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Summary:
# When run, predictor_client reads in a csv testset file from the command line.
# The test set are yelp reviews, and using trained LR, Perceptron, Nearest_Centroid, SVM
# models, will make a prediction of what star rating the user gave their review. 

# This implementation focuses around the text of the reviews.
# The text is stemmed and then normalized based on the training set. 
# Trained LR, Perceptron, Nearest_Centroid, SVM models are then used to predict results
# and the final result is decided through ensemble voting. 


# In[7]:


import pandas as pd
import nltk as nltk
import numpy as np 
import pickle
import csv
import sys
import time;
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


    # In[8]:
    
def main(argv):    
    # Read test data from json file passed through command line. 
    # Text is collected and stored in corpus. 
    # Variables: 
    #     Results = the read json file
    #     Corpus  = a list of text from the reviews
    #result = pd.read_json('../data/data_test_wo_label.json')
    result = pd.read_json(open(sys.argv[1]))
    corpus = []
    for index, row in result.iterrows():
        corpus.append(row['text'])
    
    
    # In[9]:
    
    
    # Data preprocessing: 
    # Take list of strings represending test text, and convert words to a 
    # stemmed version of that word (i.e. run, running runs -> run
    #     test_stemmed = list of stemmed test review text
    ps = PorterStemmer()
    test_stemmed = []
    for text in corpus:
        words = word_tokenize(text)
        new_text = ""
        for word in words:
            stemmed = ps.stem(word)
            new_text = new_text + " " + stemmed
        test_stemmed.append(new_text)
    
    
    # In[10]:
    
    
    # Unload pickled vectorized with prelearned exgtracted features from training.
    # Load the object from that file into vectorizer
    fileObject = open("pickled_feature_vectorizer",'rb')
    vectorizer = pickle.load(fileObject)  
    fileObject.close()
    
    # Transform test text data based on extracted features
    feature_vector = vectorizer.transform(test_stemmed)
    
    
    # In[11]:
    
    
    # Normalize test feature vectors
    # normalized_feature_vector = tranformed feature vector
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(with_mean=False)
    sc.fit(feature_vector)
    normalized_feature_vector = sc.transform(feature_vector)  
    
    
    # In[12]:
    
    
    testset = normalized_feature_vector
    # unload pickled learned models LR, Perceptron, Nearest_Centroid, SVM
    fileObject = open("pickled_logistic_regression_model",'rb')
    log_reg = pickle.load(fileObject)
    fileObject.close()
    
    fileObject = open("pickled_perceptron",'rb')
    perceptron = pickle.load(fileObject)
    fileObject.close()
    
    fileObject = open("pickled_nearest_centroid",'rb')
    nc = pickle.load(fileObject)
    fileObject.close()
    
    fileObject = open("pickled_svm", 'rb')
    svm = pickle.load(fileObject)
    fileObject.close()
    
    # Make predictions of each model using the testset
    predictions_LR = log_reg.predict(testset)
    predictions_p = perceptron.predict(testset)
    predictions_nc = nc.predict(testset)
    predictions_svm = svm.predict(testset)
    
    # Append the predictions into labels
    # labels = list of predictions from all models
    labels = []
    labels.append(predictions_LR)
    labels.append(predictions_p)
    labels.append(predictions_nc)
    labels.append(predictions_svm)
    
    
    # In[15]:
    
    
    # voting/ensemble method
    # Vote on the final prediction using the results predicted by all models,
    # then write the final predictions into a csv file.
    # final_prediction = list of final predictions
    final_predictions = []
    for instance in range(labels[0].size):
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        
        # Increase count if model predicts its label
        for model in range(4):
            vote = labels[model].item(instance)
            if(vote == '1'):
                count1 = count1+1
            if(vote == '2'):
                count2 = count2+1
            if(vote == '3'):
                count3 = count3+1
            if(vote == '4'):
                count4 = count4+1
            if(vote == '5'):
                count5 = count5+1
    
        # Find the winner. If there is a tie, select the first result. 
        # Then append the winner to final_prediction
        votes = [count1, count2, count3, count4, count5]
        winner_index = 0
        for v in range(len(votes)):
            if(votes[v] > votes[winner_index]):
                winner_index = v
        final_predictions.append(float(winner_index+1))
    
    
    # In[16]:
    
    
    # Write the final_predictions into predictions.csv
    with open('./predictions_group2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Predictions"])
        for p in final_predictions:
            writer.writerow([p])


if __name__ == "__main__":
	main(sys.argv[1:])
