# -*- coding: utf-8 -*-
"""
data_preprocessing.py

Not sure what this file will look like,
but the end goal is to do any data cleaning (specifically for data concerning the text)
and picking out features.
 
This file should give in the end, a neat dataset, with each instance in the representation
we engineered and its corresponding label (star rating).
"""
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk as nltk
nltk.download('punkt')

labels = []
corpus = [] # global list of strings, only data needed for feature extraction added
corpus_stemmed = [] # global list of strings, the corpus that has been processed to reduce variants of the same word

training_corpus = []
features = ["labels"]

# Data extraction for feature extraction function
# Variables:
#	rawData = a dataframe of data and attributes from input
def extractData(rawData):
    global corpus
    global labels
    for index, row in rawData.iterrows():
        corpus.append(row['text']) 
        labels.append(row['stars'])
    print("the corpus has length: ", len(corpus))

# Data preproessing (removing things like hunt, hunting, hunted -> hunt)
def preprocessData():
    
    # Using stemming to reduce variants of the same word
    # Ex: blossom->blossom, blossomed->blossom
    global corpus_stemmed
    ps = PorterStemmer()
    for text in corpus:
       words = word_tokenize(text)
       new_text = ""
       for word in words:
           stemmed = ps.stem(word)
           new_text = new_text + " " + stemmed
       corpus_stemmed.append(new_text)
       
def extractFeatures():
    # Default vectorizer, see how good model will be with these features
    global features
    global corpus_stemmed
    global training_corpus
    global labels
    vectorizer = TfidfVectorizer(min_df = 2, max_df = 0.5)#, ngram_range=(1,2)) # following tutorial: https://www.youtube.com/watch?v=7YacOe4XwhY
    vectorizer.fit_transform(corpus_stemmed)
    features.append(vectorizer.get_feature_names())
    
    # Turn each text review into a vector in terms of the features extracted
    index = 0
    for text in corpus_stemmed:
        document = []
        document.append(text)
        feature_vector = vectorizer.transform(document)
        feature_list = [labels[index]]
        for each in feature_vector:
            feature_list.append(each)
        training_corpus.append(feature_list)
        index = index + 1
    
# This is the trainng data represented as feature vectors and corresponding labels
# Will produce a .csv file called trainingData.csv
def outputTrainingData():
    global features
    global training_corpus
    data_csv = [] #should be a list of lists, each list element is a row for csv
    #print(len(features), "should equal", len(training_corpus[0]))
    data_csv.append(features)
    for instance in training_corpus:
        data_csv.append(instance)
    myFile = open('trainingData.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data_csv)
    print("type of features should be a list: ", type(features), " and type of training corpus should be a list of lists: ", type(training_corpus[0]))
    print("Done writing instances to trainingData.csv")
    
def main(argv):
    # Parse json file 
    result = pd.read_json('../data/data_train.json')
    
    extractData(result)
    preprocessData()
    extractFeatures()
    outputTrainingData()
    
if __name__ == "__main__":
	main(sys.argv[1:])