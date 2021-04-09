import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split


class SpamFilter:

    def __init__(self):
        self.vocabulary = {}

    def fit(self, x, y):
        self.x = [t.translate(str.maketrans("", "",
                                            string.punctuation)) for t in x]
        self.y = [cls for cls in y]
        self.length = len(self.x)

    def train(self):
        self.create_vocabulary()
        self.calculate_weights()

    def create_vocabulary(self):
        for i in range(len(self.x)):
            for word in self.x[i].split(sep=" "):
                if word not in self.vocabulary:
                    self.vocabulary[word] = {"ham": 0, "spam": 0}
                self.vocabulary[word][self.y[i]] += 1

    def calculate_weights(self):
        self.weights = {}
        self.spams = len([spam for spam in self.y if spam == "spam"])
        self.hams = len([ham for ham in self.y if ham == "ham"])
        for word, count in self.vocabulary.items():
            pr_w_s = count["spam"] / \
                self.spams if count["spam"] != 0 else 0.000000001
            pr_w_h = count["ham"] / \
                self.hams if count["ham"] != 0 else 0.000000001
            self.pr_s = self.spams / self.length
            self.pr_h = self.hams / self.length
            self.weights[word] = (pr_w_s) / (pr_w_s + pr_w_h)

    def predict(self, email):
        predictions = []
        for i in range(len(email)):
            classifier = []
            for word in email[i].split(sep=" "):
                if word in self.weights:
                    classifier.append(self.weights[word])
            if len(classifier) != 0:
                prediction = np.prod(classifier) / (
                    (
                        np.prod(classifier)
                        + (
                            (self.pr_h / self.pr_s)
                            ** (1 - len(email[i].split(sep=",")))
                        )
                        * np.prod([1 - p for p in classifier])
                    )
                )
            else:
                prediction = 0
            predictions.append("spam" if prediction >= 0.5 else "ham")
        return predictions

    def test_model(self, x, y):
        samples = [t.translate(str.maketrans(
            "", "", string.punctuation)) for t in x]
        features = [cls for cls in y]
        correct = 0
        predictions = self.predict(samples)
        length = len(predictions)
        for i in range(length):
            if predictions[i] == features[i]:
                correct += 1
                print("Accuracy: {0} %\nCorrect: {1}\nTest data samples: {2}\n".format(
                    (correct / length) * 100, correct, length
                )
                )


emails = pd.read_csv("spamdb.csv", sep=",", usecols=[0, 1], encoding="latin-1")
text_train, text_test, class_train, class_test = train_test_split(
    emails["text"], emails["class"], test_size=0.20,
    train_size=0.80
)
sf = SpamFilter()
sf.__init__()
sf.fit(text_train, class_train)
sf.train()
sf.test_model(text_test, class_test)
