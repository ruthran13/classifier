import os
import nltk
# from nltk.classify import SklearnClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.svm import SVC
from nltk.metrics.scores import (precision, recall)

file_dir = os.path.dirname(__file__)
path = 'files/classification'
abs_file_path = os.path.join(file_dir, path)
f = open(abs_file_path, "r")

training_data = []
num = 0
filelines = f.read().splitlines()

for index, line in enumerate(filelines):
    data = ()
    obj = {"sentence":""}
    line = line.strip()
    if not line:
        continue
    if line.startswith("#"):
        continue
    if num==0:
        obj["sentence"]=line
        data = data + (obj,)
        data = data + (filelines[index+1],)
        training_data.append(data)
    elif num==2:
        num = -1
        data = ()
    num += 1

training_set = training_data[:100]
testing_set  = training_data[100:]

def Naive_Bayes(sentence):
    classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
    print classifier.classify({"sentence":sentence})
    # print (sorted(classifier.labels()))
    print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    print ("Classifier precision percent:", precision(training_set,testing_set) * 100)

    p = precision(training_set, testing_set)
    r = recall(training_set, testing_set)
    alpha = 0.5

    if p is None or r is None:
        print None
    elif p == 0 or r == 0:
        print 0
    else:
        print 1.0 / (alpha / p + (1 - alpha) / r)

def Decision_Tree_Classifier(sentence):
    classifier = nltk.DecisionTreeClassifier.train(training_set)
    print classifier.classify({"sentence":sentence})
    print ("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

def Sklearn_Classifier():
    from sklearn.svm import LinearSVC
    from nltk.classify.scikitlearn import SklearnClassifier
    classifier = SklearnClassifier(LinearSVC())

Naive_Bayes("What are the models available at http://www.kaymu.lk?")
Decision_Tree_Classifier("What are the models available at http://www.kaymu.lk?")


