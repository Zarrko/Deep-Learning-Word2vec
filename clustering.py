from sklearn.cluster import KMeans
import time
from train_appointments import model
from gensim.models import Word2Vec
import reprlib


# load model from bag of words
model = Word2Vec.load("300ftures_40minWord_10ctx")

start = time.time()

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] // 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start

print ("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.wv.index2word, idx ))

# Find all of the words for that cluster number, and print them out
words = []

# print the first 10 clusters
for cluster in range(0,10):
    # Print the cluster number
    print ("\nCluster %d" % cluster)

    for key, item in word_centroid_map.items():
        if item == cluster:
            words.append(key)
    print(words)

with open("cluster.txt", 'w') as txt_file:
     txt_file.write('\n'.join(words))
    txt_file.close()

# train predictive model 




