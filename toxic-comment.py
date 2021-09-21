#!/usr/bin/env python
# coding: utf-8

# libraries
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# READING THE CSV TRAINING AND TEST FILES
training = pd.read_csv("../input/toxic-comments/train.csv")
test = pd.read_csv("../input/toxic-comments/test.csv")


# **DATA PREPROCESSING**
# 

# DATA PREPROCESSING -- TRAINING DATA
# remove blank rows in data (if any)
training['comment_text'].dropna(inplace = True)

# change all text to lowercase
training['comment_text'] = [comment.lower() for comment in training['comment_text']]

# tokenization
training['comment_text'] = [word_tokenize(comment) for comment in training['comment_text']]

# remove stop-words, non-numeric, and perform Word stemming/lemming
# adding positive tags to help WordNetLemmatizer understand if a word is a noun/verb/adj
tag_map = defaultdict(lambda : wn.NOUN) 
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
# for loop to remove stop-words, non-numeric, and perform Word stemming/lemming
n = 0
for index,comment in enumerate(training['comment_text']):
    final_words = [] # empty list to store words 
    word_lemmatized = WordNetLemmatizer() # creating python object for lemmatization
    # pos_tag function provides info about whether word is noun, verb, adj, etc.
    for word, tag in pos_tag(comment):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word,tag_map[tag[0]])
            final_words.append(word_final)
    # will store final processed words in a new column in the data frame
    training.loc[index,'processed_comments'] = str(final_words)
    n += 1
    # print(n) -- just to keep track of how long until it's done running 


# DATA PREPROCESSING - TEST DATA
# remove blank rows in data (if any)
test['comment_text'].dropna(inplace = True)

# change all text to lowercase
test['comment_text'] = [comment.lower() for comment in test['comment_text']]

# tokenization
test['comment_text'] = [word_tokenize(comment) for comment in test['comment_text']]

# remove stop-words, non-numeric, and perform Word stemming/lemming
# adding positive tags to help WordNetLemmatizer understand if a word is a noun/verb/adj
# (commented out the stuff below because it's in the cell above)
#tag_map = defaultdict(lambda : wn.NOUN) 
#tag_map['J'] = wn.ADJ
#tag_map['V'] = wn.VERB
#tag_map['R'] = wn.ADV
# for loop to remove stop-words, non-numeric, and perform Word stemming/lemming
n = 0
for index,comment in enumerate(test['comment_text']):
    final_words = [] # empty list to store words 
    word_lemmatized = WordNetLemmatizer() # creating python object for lemmatization
    # pos_tag function provides info about whether word is noun, verb, adj, etc.
    for word, tag in pos_tag(comment):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word,tag_map[tag[0]])
            final_words.append(word_final)
    # will store final processed words in a new column in the data frame
    test.loc[index,'processed_comments'] = str(final_words)
    n += 1
    # print(n) -- just to keep track of how long until it's done running 
test


# WORD VECTORIZATION
# using tf-idf - first build the model
tfidf = TfidfVectorizer() # maybe put max_features=5000 if SVM isn't accurate enough
# fitting the model
tfidf.fit(training['processed_comments'])
# performing word vectorization 
training_tfidf = tfidf.transform(training['processed_comments'])
test_tfidf = tfidf.transform(test['processed_comments'])
# something i'm interested in-- can see the vocabulary learned from the processed comments
print(tfidf.vocabulary_)



# ENCODING
# need to transform categorical binary labels into something the model will understand
# transforming all of the training labels
encoder = LabelEncoder()

# label 1 - toxic
toxic_training = training['toxic']
toxic_traning = encoder.fit_transform(toxic_training)

# label 2 - severe toxic
severe_toxic_training = training['severe_toxic']
severe_toxic_training = encoder.fit_transform(severe_toxic_training)

# label 3 - obscene
obscene_training = training['obscene']
obscene_training = encoder.fit_transform(obscene_training)

# label 4 - threat
threat_training = training['threat']
threat_training = encoder.fit_transform(threat_training)

# label 5 - insult
insult_training = training['insult']
insult_training = encoder.fit_transform(insult_training)

# label 6 - identity hate
identity_hate_training = training['identity_hate']
identity_hate_training = encoder.fit_transform(identity_hate_training)


# SVM Algorithm - six different labels, need six different SVM models
# setting kernel to linear because it's a large data set

# ideally would perform cross validation to calculate C but there's no way 
# I would be able to do that on my computer within a realistic time frame
# so I'm starting with C = 1 and I will test a larger value of C later on

# model 1 - label 1, toxic comments
SVM_toxic_1 = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto') 
SVM_toxic_1.fit(training_tfidf, toxic_training)
toxic_predictions_1 = SVM_toxic_1.predict(test_tfidf)

# model 2 - label 2, severe toxic comments
SVM_severe_toxic_1 = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto') 
SVM_severe_toxic_1.fit(training_tfidf, severe_toxic_training)
severe_toxic_predictions_1 = SVM_severe_toxic_1.predict(test_tfidf)

# model 3 - label 3, obscene comments
SVM_obscene_1 = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto') 
SVM_obscene_1.fit(training_tfidf, obscene_training)
obscene_predictions_1 = SVM_obscene_1.predict(test_tfidf)

# model 4 - label 4, threatening comments
SVM_threat_1 = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto') 
SVM_threat_1.fit(training_tfidf, threat_training)
threat_predictions_1 = SVM_threat_1.predict(test_tfidf)

# model 5 - label 5, insulting comments
SVM_insult_1 = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto') 
SVM_insult_1.fit(training_tfidf, insult_training)
insult_predictions_1 = SVM_insult_1.predict(test_tfidf)

# model 6 - label 6, identity hate comments
SVM_identity_hate_1 = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto') 
SVM_identity_hate_1.fit(training_tfidf, identity_hate_training)
identity_hate_predictions_1 = SVM_identity_hate_1.predict(test_tfidf)


# data frame of predictions
data_1 = {
    'test_comments' : test['comment_text'],
    'toxic' : toxic_predictions_1,
    'severe_toxic' : severe_toxic_predictions_1, 
    'obscene' : obscene_predictions_1,
    'threat': threat_predictions_1,
    'insult' : insult_predictions_1,
    'identity_hate': identity_hate_predictions_1
}
results_1 = pd.DataFrame(data_1)


# GETTING TEST LABELS TO CHECK SVM ACCURACY
# test labels (available since competition is over)
test_labels = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip') 
test_labels.dropna(inplace = True) # dropping null values just in case

# negative 1s means that the label was not used for scoring-- so I will drop them from the data frame
# but first need indices of the rows deleted from this data frame so that the same rows 
# can be deleted from the results data frame
# since it's negative 1 for all entries of a row not used, only need to check 1 variable
rows2drop = test_labels[test_labels.toxic == -1 ].index
test_labels.drop(index = rows2drop)
results_1.drop(index = rows2drop)


# In[ ]:


# getting test labels & encoding
toxic_test = test_labels['toxic']
toxic_test = encoder.fit_transform(toxic_test)

severe_toxic_test = test_labels['severe_toxic']
severe_toxic_test = encoder.fit_transform(severe_toxic_test)

obscene_test = test_labels['obscene']
obscene_test = encoder.fit_transform(obscene_test)

threat_test = test_labels['threat']
threat_test = encoder.fit_transform(threat_test)

insult_test = test_labels['insult']
insult_test = encoder.fit_transform(insult_test)

identity_hate_test = test_labels['identity_hate']
identity_hate_test = encoder.fit_transform(identity_hate_test)


# CHECKING SVM ACCURACY
# accuracy_score(predictions, test)
# toxic label accuracy
SVM_toxic_accuracy_1 = accuracy_score(results_1['toxic'], toxic_test)
SVM_toxic_accuracy_1 



# severe toxic label accuracy
SVM_severe_toxic_accuracy_1 = accuracy_score(results_1['severe_toxic'], severe_toxic_test)
SVM_severe_toxic_accuracy_1



# obscene label accuracy
SVM_obscene_accuracy_1 = accuracy_score(results_1['obscene'], obscene_test)
SVM_obscene_accuracy_1



# threat label accuracy
SVM_threat_accuracy_1 = accuracy_score(results_1['threat'], threat_test)
SVM_threat_accuracy_1


# insult label accuracy
SVM_insult_accuracy_1 = accuracy_score(results_1['insult'], insult_test)
SVM_insult_accuracy_1



# identity hate label accuracy
SVM_identity_hate_accuracy_1 = accuracy_score(results_1['identity_hate'], identity_hate_test)
SVM_identity_hate_accuracy_1



# making a pie chart for test labels
labels = 'toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate'

# need to get sizes of each 
toxic_bool_test = test_labels.apply(lambda x: True if x['toxic'] == 1 else False, axis = 1)
toxic_count_test = len(toxic_bool_test[toxic_bool_test == True].index) # toxic size

severe_toxic_bool_test = test_labels.apply(lambda x: True if x['severe_toxic'] == 1 else False, axis = 1)
severe_toxic_count_test = len(severe_toxic_bool_test[severe_toxic_bool_test == True].index) # severe toxic size

obscene_bool_test = test_labels.apply(lambda x: True if x['obscene'] == 1 else False, axis = 1)
obscene_count_test = len(obscene_bool_test[obscene_bool_test == True].index) # obscene size

threat_bool_test = test_labels.apply(lambda x: True if x['threat'] == 1 else False, axis = 1)
threat_count_test = len(threat_bool_test[threat_bool_test == True].index) # threat size

insult_bool_test = test_labels.apply(lambda x: True if x['insult'] == 1 else False, axis = 1)
insult_count_test = len(insult_bool_test[insult_bool_test == True].index) # insult size

identity_hate_bool_test = test_labels.apply(lambda x: True if x['identity_hate'] == 1 else False, axis = 1)
identity_hate_count_test = len(identity_hate_bool_test[identity_hate_bool_test == True].index) # identity hate size

sizes_pie_test = [toxic_count_test, severe_toxic_count_test, obscene_count_test, 
                  threat_count_test, insult_count_test, identity_hate_count_test]

# choosing colors
colors_test = ['forestgreen', 'limegreen', 'mediumaquamarine', 'aquamarine', 'lightseagreen', 'lightcyan']

# plotting the pie chart
patches, text = plt.pie(sizes_pie_test, colors=colors_test)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()



# making a pie chart for C = 1 only (since C = 100 returns rly similar results)

# getting sizes
toxic_bool_1 = results_1.apply(lambda x: True if x['toxic'] == 1 else False, axis = 1)
toxic_count_1 = len(toxic_bool_1[toxic_bool_1 == True].index) # toxic size

severe_toxic_bool_1 = results_1.apply(lambda x: True if x['severe_toxic'] == 1 else False, axis = 1)
severe_toxic_count_1 = len(severe_toxic_bool_1[severe_toxic_bool_1 == True].index) # severe toxic size

obscene_bool_1 = results_1.apply(lambda x: True if x['obscene'] == 1 else False, axis = 1)
obscene_count_1 = len(obscene_bool_1[obscene_bool_1 == True].index) # obscene size

threat_bool_1 = results_1.apply(lambda x: True if x['threat'] == 1 else False, axis = 1)
threat_count_1 = len(threat_bool_1[threat_bool_1 == True].index) # threat size

insult_bool_1 = results_1.apply(lambda x: True if x['insult'] == 1 else False, axis = 1)
insult_count_1 = len(insult_bool_1[insult_bool_1 == True].index) # insult size

identity_hate_bool_1 = results_1.apply(lambda x: True if x['identity_hate'] == 1 else False, axis = 1)
identity_hate_count_1 = len(identity_hate_bool_1[identity_hate_bool_1 == True].index) # identity hate size

sizes_pie_1 = [toxic_count_1, severe_toxic_count_1, obscene_count_1, 
                  threat_count_1, insult_count_1, identity_hate_count_1]

# choosing colors
colors_1 = ['cornflowerblue', 'darkviolet', 'midnightblue', 'blue', 'slateblue', 'lightsteelblue']
# plotting the pie chart
patches, text = plt.pie(sizes_pie_1, colors=colors_1)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


# The following is the output for C = 100.

# trying C = 100

# model 1 - label 1, toxic comments
SVM_toxic_100 = svm.SVC(C=100, kernel='linear', degree=3, gamma='auto') 
SVM_toxic_100.fit(training_tfidf, toxic_training)
toxic_predictions_100 = SVM_toxic_100.predict(test_tfidf)

# model 2 - label 2, severe toxic comments
SVM_severe_toxic_100 = svm.SVC(C=100, kernel='linear', degree=3, gamma='auto') 
SVM_severe_toxic_100.fit(training_tfidf, severe_toxic_training)
severe_toxic_predictions_100 = SVM_severe_toxic_100.predict(test_tfidf)

# model 3 - label 3, obscene comments
SVM_obscene_100 = svm.SVC(C=100, kernel='linear', degree=3, gamma='auto') 
SVM_obscene_100.fit(training_tfidf, obscene_training)
obscene_predictions_100 = SVM_obscene_100.predict(test_tfidf)

# model 4 - label 4, threatening comments
SVM_threat_100 = svm.SVC(C=100, kernel='linear', degree=3, gamma='auto') 
SVM_threat_100.fit(training_tfidf, threat_training)
threat_predictions_100 = SVM_threat_100.predict(test_tfidf)

# model 5 - label 5, insulting comments
SVM_insult_100 = svm.SVC(C=100, kernel='linear', degree=3, gamma='auto') 
SVM_insult_100.fit(training_tfidf, insult_training)
insult_predictions_100 = SVM_insult_100.predict(test_tfidf)

# model 6 - label 6, identity hate comments
SVM_identity_hate_100 = svm.SVC(C=100, kernel='linear', degree=3, gamma='auto') 
SVM_identity_hate_100.fit(training_tfidf, identity_hate_training)
identity_hate_predictions_100 = SVM_identity_hate_100.predict(test_tfidf)



# data frame of predictions
data_100 = {
    'test_comments' : test['comment_text'],
    'toxic' : toxic_predictions_100,
    'severe_toxic' : severe_toxic_predictions_100, 
    'obscene' : obscene_predictions_100,
    'threat': threat_predictions_100,
    'insult' : insult_predictions_100,
    'identity_hate': identity_hate_predictions_100
}
results_100 = pd.DataFrame(data_100)
results_100


# CHECKING SVM ACCURACY
results_100.drop(index = rows2drop) # making sure the data frame matches the number of test labels
# accuracy_score(predictions, test)
# toxic label accuracy
SVM_toxic_accuracy_100 = accuracy_score(results_100['toxic'], toxic_test)
SVM_toxic_accuracy_100




# severe toxic label accuracy
SVM_severe_toxic_accuracy_100 = accuracy_score(severe_toxic_predictions_100, severe_toxic_test)
SVM_severe_toxic_accuracy_100



# obscene label accuracy
SVM_obscene_accuracy_100 = accuracy_score(obscene_predictions_100, obscene_test)
SVM_obscene_accuracy_100


# threat label accuracy
SVM_threat_accuracy_100 = accuracy_score(threat_predictions_100, threat_test)
SVM_threat_accuracy_100



# insult label accuracy
SVM_insult_accuracy_100 = accuracy_score(insult_predictions_100, insult_test)
SVM_insult_accuracy_100


# identity hate label accuracy
SVM_identity_hate_accuracy_100 = accuracy_score(identity_hate_predictions_100, identity_hate_test)
SVM_identity_hate_accuracy_100


