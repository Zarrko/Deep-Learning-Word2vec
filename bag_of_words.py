from appointments_text import appointments
from appointments_text import split_appointments
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import nltk.data
import logging
from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# split appointments into two
train, test =split_appointments(appointments)

print(len(train))
print(len(test))


def appointment_to_wordlist(appointment, remove_stopwords=False):
     # convert a document to a sequence of words
     # Return a list of words

     # Remove HTML
     appointment_text = BeautifulSoup(appointment, "lxml").get_text()

     # Remove Non Letters
     appointment_text = re.sub("[^a-zA-Z]"," ", appointment_text)

     # convert to lower case and split them
     words = appointment_text.lower().split()

     #Optionally remove stop words (false by default)
     if remove_stopwords:
         stops = set(stopwords.words("english"))
         words = [w for w in words if not w in stops]

     return words

# Load punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Define a function to split an appointment text into parsed sentences
def review_to_sentences( appointment, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words

    # NLTK Tokenizer to split paragraphs into sentences
    raw_sentences = tokenizer.tokenize(appointment.strip())

    # loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # if a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # otherwise call appointment to wordlist
            sentences.append(appointment_to_wordlist(raw_sentence, remove_stopwords))

    # return a list of sentences
    return sentences


# an empty list of sentences
sentences = []
print("Parsing Sentences from Training set")
for appointment in train:
    sentences += review_to_sentences(appointment, tokenizer)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 5       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print("Training Model")

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

# model name
model_name = "300ftures_40minWord_10ctx"
model.save(model_name)

# train a classifier





