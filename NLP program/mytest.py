#importing essential libraries
import tensorflow as tf
from tensorflow import keras as keras
#tensorflow and keras are needed to create model
import pandas as pd
#pandas is needed to read and manipulate the dataset
import numpy as np
#needed for preprocessing 

#reading in the datasets
myreviews = pd.read_csv("imdb_reviews.csv")
testreviews = pd.read_csv("test_reviews.csv")

#reading in file that contains dataset of keywords to help with preprocessing
wordindex = pd.read_csv("word_indexes.csv")
#converting the stuff in the wordindex file into a dictionary dataset
wordindex = dict(zip(wordindex.Words, wordindex.Indexes))
wordindex["<PAD>"] = 0
wordindex["<START"] = 1
wordindex["<UNK>"] = 2
wordindex["<UNUSED>"] = 3

#creating function to encoder the reviews with the parameters being the text
def reviewencoder(text):
    arr = [wordindex[word] for word in text]
    return arr

#splitting the data set of the reviews 
trainingdata, traininglabels = myreviews["Reviews"], myreviews["Sentiment"]
testingdata, testinglabels = testreviews["Reviews"], testreviews["Sentiment"]

#tokenizing the strings in the review areas so that we can map the words
trainingdata = trainingdata.apply(lambda review:review.split()) 
testingdata = testingdata.apply(lambda review:review.split()) 

#using the reviewencoder function to put the indexes into an easier form for preprocessing
#apply method lets you appy variable to a function
trainingdata = trainingdata.apply(reviewencoder)
testingdata = testingdata.apply(reviewencoder)

#creating functions to give values to the sentimenmts
def sentimentencoder(sentiment):
    #this will go through sentiments in the files and return values based on the string values
    if sentiment == "positive":
        return 1
    if sentiment == "negative":
        return 0

#giving values to the sentiments/labels
traininglabels = traininglabels.apply(sentimentencoder)
testinglabels = testinglabels.apply(sentimentencoder)

#adding a limit to how many indexes each review has to remove any kind of ambiguity in our input for the data
#the length limit is going to be at 500 indexes, any review with less than 500 will have 0s added until cap is met
trainingdata = keras.preprocessing.sequence.pad_sequences(trainingdata, value = wordindex["<PAD>"], padding = "post", maxlen = 500)
testingdata = keras.preprocessing.sequence.pad_sequences(trainingdata, value = wordindex["<PAD>"], padding = "post", maxlen = 500)

#creating the model architetecture
#first layer is word embedding layer, word embedding is needed to have model understand connection between similar words
#Global average will help return probability of review being positive or negative
model = keras.Sequential([keras.layers.Embedding(10000, 16, input_length= 500), keras.layers.GlobalAveragePooling1D(), keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#establishing history variable to train the model
history = model.fit(trainingdata, traininglabels, epochs=30, batch_size=512, validation_data=(testingdata,testinglabels))

loss,accuracy = model.evaluate(testingdata, testinglabels) 



