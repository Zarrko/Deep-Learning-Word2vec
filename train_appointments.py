from gensim.models import Word2Vec

from gensim.models import word2vec
import numpy as np
from bag_of_words import train
from bag_of_words import test
from bag_of_words import appointment_to_wordlist
from bag_of_words import num_features
from gensim.models import keyedvectors


# load model from bag of words
model = Word2Vec.load("300ftures_40minWord_10ctx")

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph

    #pre-initialize an empty numpy array
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# Given a set of reviews (each one a list of words), calculate
# the average feature vector for each one and return a 2D numpy array
def getAvgFeatureVec(appointments, model, num_features):
    # initialize counter
    counter = 0

    # Preaallocate a 2D numpy array for speed
    reviewFeatureVecs = np.zeros((len(appointments), num_features), dtype="float32")

    # Loop through the reviews
    for appointment in appointments:
        # Print a status message every 1000th review
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(appointments)))

        # average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(appointment, model, num_features)

        # Increment counter
        counter = counter + 1
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_appointments = []
for appointment in train:
    clean_train_appointments.append(appointment_to_wordlist(appointment, remove_stopwords=True))

trainDataVecs = getAvgFeatureVec(clean_train_appointments, model, num_features)

print("Creating Average Feature vecs for test reviews")
clean_test_appointments = []
for appointment in test:
    clean_test_appointments.append(appointment_to_wordlist(appointment, remove_stopwords=True))

testDataVecs = getAvgFeatureVec(clean_test_appointments, model, num_features)








