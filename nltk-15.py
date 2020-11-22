#Naive Bayes Classifier with NLTK

import nltk 
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

documents = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)    ]

# documents = []
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)), category)

random.shuffle(documents)

#print(documents[1])

all_words = []

for w in movie_reviews.words():
    w = w.lower()
    all_words.append(w)

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["stupid"])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev,category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo accuracy percent :", (nltk.classify.accuracy(classifier, testing_set)) *100)
classifier.show_most_informative_features(15) #show most papular words from both sides

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Naive Bayes Algo accuracy percent :", (nltk.classify.accuracy(MNB_classifier, testing_set)) *100)


# GaussianNB and BernoulliNB
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print("GaussianNB_classifier Naive Bayes Algo accuracy percent :", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) *100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Naive Bayes Algo accuracy percent :", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) *100)



