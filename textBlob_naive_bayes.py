import os
import nltk
# nltk.download('punkt')
from textblob.classifiers import NaiveBayesClassifier

file_dir = os.path.dirname(__file__)
path = 'files/classification'
abs_file_path = os.path.join(file_dir, path)
f = open(abs_file_path, "r")

training_data = []
num = 0
filelines = f.read().splitlines()

for index, line in enumerate(filelines):
    data = ()
    line = line.strip()
    if not line:
        continue
    if line.startswith("#"):
        continue
    if num==0:
        data = data + (line,)
        data = data + (filelines[index+1],)
        training_data.append(data)
    elif num==2:
        num = -1
        data = ()
    num += 1

cl = NaiveBayesClassifier(training_data[:100])

def classify (sentence):
    print(cl.classify(sentence))

def accuracy():
    print cl.accuracy(training_data[100:])

# classify()
accuracy()