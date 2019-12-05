# !python -m spacy download en_core_web_sm
import spacy
import en_core_web_sm
import numpy as np
import pandas as pd
from numpy import loadtxt
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from keras.models import Model
import pandas as pd

nlp = en_core_web_sm.load()


def model():
    # Define two input layers
    input = Input((3, 96))
    #vector_input = Input((6,))

    # Convolution + Flatten for the image
    #conv_layer = Conv2D(32, (3, 3))(image_input)
    #flat_layer = Flatten()(conv_layer)

    # Concatenate the convolutional features and the vector input
    #concat_layer = Concatenate()([vector_input, flat_layer])
    outputa = Dense(64)(input)
    outputb = Dense(32)(outputa)
    outputb = Flatten()(outputb)
    outputc = Dense(2, activation='sigmoid')(outputb)

    # define a model with a list of two inputs
    model = Model(inputs=[input], outputs=outputc)
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
        doc_q = "{} {} {} {}".format(
            doc["InputSentence1"], doc["InputSentence2"], doc["InputSentence3"], doc["InputSentence4"])
        doc_data["input_sentence"] = nlp(doc_q).vector
        doc_data["output_one"] = nlp(doc["RandomFifthSentenceQuiz1"]).vector
        doc_data["output_two"] = nlp(doc["RandomFifthSentenceQuiz2"]).vector
        doc_data["label"] = doc["AnswerRightEnding"]
        dict_data.append(doc_data)
    return(dict_data)


def convert_to_input(dict_data):
    X = []
    Y = []
    for v in dict_data:
        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(v["output_one"])
        finaler.append(v["output_two"])
        label = v["label"]
        if label == 1:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
        finaler = np.array(finaler)
        X.append(finaler)

        finaler = []
        finaler.append(v["input_sentence"])
        finaler.append(v["output_two"])
        finaler.append(v["output_one"])
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
