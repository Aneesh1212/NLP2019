# !python -m spacy download en_core_web_sm
import spacy
import en_core_web_sm
import numpy as np
import pandas as pd
from numpy import loadtxt
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from keras.models import Model
import pandas as pd
from textblob import TextBlob
nlp = en_core_web_sm.load()


def ngram_generate(csv_path):
    vals = []
    csv = pd.read_csv(csv_path)
    vals.append(csv["InputSentence1"].tolist())
    vals.append(csv["InputSentence2"].tolist())
    vals.append(csv["InputSentence3"].tolist())
    vals.append(csv["InputSentence4"].tolist())
    vals.append(csv["RandomFifthSentenceQuiz1"].tolist())
    vals.append(csv["RandomFifthSentenceQuiz2"].tolist())
    vectorizer = TfidfVectorizer(stop_words="english",
                                 ngram_range=(1, 2))
    vals = np.array(np.concatenate(vals).ravel().tolist())
    training_features = vectorizer.fit_transform(vals)
    return(vectorizer)


l = ngram_generate('/content/train.csv')


def read_csv(csv_path):
    return(pd.read_csv(csv_path))


def generate_embeddings_per_document(document):
    return(nlp(document))


def preprocess(data, l):
    dict_data = []
    for i, document in enumerate(data.iterrows()):
        doc_data = {"input_sentence": [], }
        doc = document[1]
        doc_q = "{} {} {} {}".format(
            doc["InputSentence1"], doc["InputSentence2"], doc["InputSentence3"], doc["InputSentence4"])
        doc_data["input_sentence"] = nlp(doc_q).vector
        blob_input = TextBlob(doc["InputSentence4"]
                              ).sentences[::-1][0].sentiment.polarity
        doc_data["input_sentiment"] = blob_input
        doc_data["input_bag"] = l.transform([doc_q]).toarray()

        doc_data["output_one"] = nlp(doc["RandomFifthSentenceQuiz1"]).vector
        blob_one = TextBlob(doc["RandomFifthSentenceQuiz1"]
                            ).sentences[::-1][0].sentiment.polarity
        doc_data["output_one_sentiment"] = blob_one
        doc_data["output_one_bag"] = l.transform(
            [doc["RandomFifthSentenceQuiz1"]]).toarray()
        doc_data["output_two"] = nlp(doc["RandomFifthSentenceQuiz2"]).vector
        blob_two = TextBlob(doc["RandomFifthSentenceQuiz2"]
                            ).sentences[::-1][0].sentiment.polarity
        doc_data["output_two_sentiment"] = blob_two
        doc_data["output_two_bag"] = l.transform(
            [doc["RandomFifthSentenceQuiz2"]]).toarray()

        doc_data["label"] = doc["AnswerRightEnding"]
        dict_data.append(doc_data)
    return(dict_data)


def convert_to_input(dict_data):
    X = []
    Y = []
    for v in dict_data:
        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(v["input_sentiment"])
        finaler.append(v["input_bag"])
        finaler.append(v["output_one"])
        finaler.append(v["output_one_sentiment"])
        finaler.append(v["output_one_bag"])
        finaler.append(v["output_two"])
        finaler.append(v["output_two_sentiment"])
        finaler.append(v["output_two_bag"])
        label = v["label"]
        if label == 1:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
        finaler = np.array(finaler)
        new_f = []
        for x in (finaler):
            if type(x) == float:
                zerod = np.zeros((30109, ))
                zerod[0] = x
                new_f.append(zerod)
            else:
                zerod = np.zeros((30109, ))
                for i, av in enumerate(x.flatten()):
                    zerod[i] = av
                new_f.append(zerod)
        finaler = np.array(new_f)
        X.append(finaler)
        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(v["input_sentiment"])
        finaler.append(v["input_bag"])
        finaler.append(v["output_two"])
        finaler.append(v["output_two_sentiment"])
        finaler.append(v["output_two_bag"])
        finaler.append(v["output_one"])
        finaler.append(v["output_one_sentiment"])
        finaler.append(v["output_one_bag"])
        if 3-label == 1:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
        finaler = np.array(finaler)
        new_f = []
        for x in (finaler):
            if type(x) == float:
                zerod = np.zeros((30109, ))
                zerod[0] = x
                new_f.append(zerod)
            else:
                zerod = np.zeros((30109, ))
                for i, av in enumerate(x.flatten()):
                    zerod[i] = av
                new_f.append(zerod)
        finaler = np.array(new_f)
        X.append(finaler)
    return(np.array(X), np.array(Y))


def train(csv_path):
    data = read_csv(csv_path)
    l = ngram_generate(csv_path)
    dict_data = preprocess(data, l)
    X, Y = convert_to_input(dict_data)
    model = model_generate()
    model.fit(X, Y, epochs=10)
    return(model)

# test set validation


def preprocess_test(data):
    dict_data = []
    for i, document in enumerate(data.iterrows()):
        doc_data = {"input_sentence": [], }
        doc = document[1]
        doc_data["id"] = doc["InputStoryid"]
        doc_q = "{} {} {} {}".format(
            doc["InputSentence1"], doc["InputSentence2"], doc["InputSentence3"], doc["InputSentence4"])
        doc_data["input_sentence"] = nlp(doc_q).vector
        blob_input = TextBlob(doc["InputSentence4"]
                              ).sentences[::-1][0].sentiment.polarity
        doc_data["input_sentiment"] = blob_input
        doc_data["input_bag"] = l.transform([doc_q]).toarray()

        # doc["RandomFifthSentenceQuiz1"]
        doc_data["output_one"] = nlp(doc["RandomFifthSentenceQuiz1"]).vector
        blob_one = TextBlob(doc["RandomFifthSentenceQuiz1"]
                            ).sentences[::-1][0].sentiment.polarity
        doc_data["output_one_sentiment"] = blob_one
        doc_data["output_one_bag"] = l.transform(
            [doc["RandomFifthSentenceQuiz1"]]).toarray()
        doc_data["output_two"] = nlp(doc["RandomFifthSentenceQuiz2"]).vector
        blob_two = TextBlob(doc["RandomFifthSentenceQuiz2"]
                            ).sentences[::-1][0].sentiment.polarity
        doc_data["output_two_sentiment"] = blob_two
        doc_data["output_two_bag"] = l.transform(
            [doc["RandomFifthSentenceQuiz2"]]).toarray()
        dict_data.append(doc_data)
    return(dict_data)


def convert_to_input_test(dict_data):
    X = []
    Y = []
    for v in dict_data:
        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(v["input_sentiment"])
        finaler.append(v["input_bag"])
        finaler.append(v["output_one"])
        finaler.append(v["output_one_sentiment"])
        finaler.append(v["output_one_bag"])
        finaler.append(v["output_two"])
        finaler.append(v["output_two_sentiment"])
        finaler.append(v["output_two_bag"])
        Y.append(v["id"])
        finaler = np.array(finaler)
        new_f = []
        for x in (finaler):
            if type(x) == float:
                zerod = np.zeros((30109, ))
                zerod[0] = x
                new_f.append(zerod)
            else:
                zerod = np.zeros((30109, ))
                for i, av in enumerate(x.flatten()):
                    zerod[i] = av
                new_f.append(zerod)
        finaler = np.array(new_f)
        X.append(finaler)
    return(np.array(X), np.array(Y))
