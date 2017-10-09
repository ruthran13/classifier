import os
import nltk
from nltk import collections
from nltk.metrics.scores import (precision, recall, f_measure)
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk.classify import maxent
from nltk.classify import NaiveBayesClassifier

file_dir = os.path.dirname(__file__)
path = 'files/classification'
abs_file_path = os.path.join(file_dir, path)
f = open(abs_file_path, "r")

training_data = []
sentence = []
intent = []
count = []
num = 0
cou = 0
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
        sentence.append(line)
        intent.append(filelines[index+1])
        count.append(cou)
        cou += 1
        obj["sentence"]=line
        data = data + (obj,)
        data = data + (filelines[index+1],)
        training_data.append(data)
    elif num==2:
        num = -1
        data = ()
    num += 1

cutoff = len(training_data)*3/4
training_set = training_data[:cutoff]
testing_set  = training_data[cutoff:]

def textDict (text):
    return dict([(word,True) for word in text])

def calculatePrecision(classifier):
    feats = [(textDict(sentence[x]), intent[x]) for x in count]

    trainfeats = feats[:cutoff]
    testfeats = feats[cutoff:]
    Classifier = classifier.train(trainfeats)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feat, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = Classifier.classify(feat)
        testsets[observed].add(i)

    print 'model precision:', precision(refsets['model'], testsets['model'])
    print 'memory precision:', precision(refsets['memory'], testsets['memory'])
    print 'onlineStore precision:', precision(refsets['onlineStore'], testsets['onlineStore'])
    print 'brand precision:', precision(refsets['brand'], testsets['brand'])

def Naive_Bayes(sentence):
    classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
    naivebayes = nltk.NaiveBayesClassifier
    calculatePrecision(naivebayes)
    # print classifier.classify({"sentence":sentence})
    # print (sorted(classifier.labels()))
    print("NaiveBayesClassifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

def Decision_Tree_Classifier(sentence):
    classifier = nltk.DecisionTreeClassifier.train(training_set)
    decisionTree = nltk.DecisionTreeClassifier
    calculatePrecision(decisionTree)
    # print classifier.classify({"sentence":sentence})
    print ("DecisionTreeClassifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

def Sklearn_Classifier(sentence):
    classifier = SklearnClassifier(BernoulliNB()).train(training_set)
    Sklearn = SklearnClassifier(BernoulliNB())
    calculatePrecision(Sklearn)
    # print classifier.classify({"sentence": sentence})
    print ("SklearnClassifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

def Maxent_Classifier(sentence):
    encoding = maxent.TypedMaxentFeatureEncoding.train(training_set, count_cutoff = 0, alwayson_features = True)
    classifier = maxent.MaxentClassifier.train(training_set, bernoulli=False, encoding=encoding, trace=0)
    Maxent = maxent.MaxentClassifier
    calculatePrecision(Maxent)
    # print classifier.classify({"sentence": sentence})
    print ("MaxentClassifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)

Naive_Bayes("What are the models available at http://www.kaymu.lk?")
Sklearn_Classifier("What are the models available at http://www.kaymu.lk?")
Decision_Tree_Classifier("Where can I get Huawei Ascend G7?")
Maxent_Classifier("which is the phone model with the price between 30000-40000?")
