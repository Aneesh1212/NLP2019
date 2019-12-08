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


def model():
    # Define two input layers
    input = Input((6,))
    #vector_input = Input((6,))

    # Convolution + Flatten for the image
    #conv_layer = Conv2D(32, (3, 3))(image_input)
    #flat_layer = Flatten()(conv_layer)

    # Concatenate the convolutional features and the vector input
    #concat_layer = Concatenate()([vector_input, flat_layer])
    outputa = Dense(256)(input)
    outputb = Dense(128)(outputa)
    outputc = Dense(64)(outputb)
    #outputc = Flatten()(outputc)
    outputd = Dense(2, activation='sigmoid')(outputc)

    # define a model with a list of two inputs
    model = Model(inputs=[input], outputs=outputd)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return(model)


def read_csv(csv_path):
    return(pd.read_csv(csv_path))


def generate_embeddings_per_document(document):
    return(nlp(document))


def preprocess(data):
    dict_data = []
    for i, document in enumerate(data.iterrows()):
        doc_data = {"input_sentence": [], }
        doc = document[1]
        # doc_q = "{} {} {} {}".format(
        #    doc["InputSentence1"], doc["InputSentence2"], doc["InputSentence3"], doc["InputSentence4"])
        doc_data["input_sentence"] = nlp(doc["InputSentence4"]).vector
        blob_input = TextBlob(doc["InputSentence4"]
                              ).sentences[::-1][0].sentiment.polarity
        doc_data["input_sentiment"] = blob_input

        # doc["RandomFifthSentenceQuiz1"]
        doc_data["output_one"] = nlp(doc["RandomFifthSentenceQuiz1"]).vector
        blob_one = TextBlob(doc["RandomFifthSentenceQuiz1"]
                            ).sentences[::-1][0].sentiment.polarity
        doc_data["output_one_sentiment"] = blob_one
        doc_data["output_two"] = nlp(doc["RandomFifthSentenceQuiz2"]).vector
        blob_two = TextBlob(doc["RandomFifthSentenceQuiz2"]
                            ).sentences[::-1][0].sentiment.polarity
        doc_data["output_two_sentiment"] = blob_two
        doc_data["label"] = doc["AnswerRightEnding"]
        dict_data.append(doc_data)
    return(dict_data)


def convert_to_input(dict_data):
    X = []
    Y = []
    for v in dict_data:
        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(np.array(v["input_sentiment"]))
        finaler.append(v["output_one"])
        finaler.append(np.array(v["output_one_sentiment"]))
        finaler.append(v["output_two"])
        finaler.append(np.array(v["output_two_sentiment"]))
        label = v["label"]
        if label == 1:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
        finaler = np.array(finaler)
        X.append(finaler)

        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(np.array(v["input_sentiment"]))
        finaler.append(v["output_two"])
        finaler.append(np.array(v["output_two_sentiment"]))
        finaler.append(v["output_one"])
        finaler.append(np.array(v["output_one_sentiment"]))
        if 3-label == 1:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
        finaler = np.array(finaler)
        X.append(finaler)
    return(np.array(X), np.array(Y))


def train(csv_path):
    data = read_csv(csv_path)
    # TODO this is redundant, just combine this line and next
    dict_data = preprocess(data)
    X, Y = convert_to_input(dict_data)
    model = model()
    model.fit(X, Y, epochs=100)
    return(model)
