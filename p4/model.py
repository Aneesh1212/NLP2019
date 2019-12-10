from data import read_csv, preprocess, convert_to_input, preprocess_test, convert_to_input_test
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
import csv

nlp = en_core_web_sm.load()


def model_generate():
    input = Input(shape=(9, 30109))
    fdf = Flatten()(input)
    outputa = Dense(512)(fdf)
    outputb = Dense(128)(outputa)
    outputc = Dense(64)(outputb)
    outputd = Dense(2, activation='sigmoid')(outputc)
    model = Model(inputs=[input], outputs=outputd)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return(model)


def model_generate_smaller():
    input = Input(shape=(9, 30109))
    fdf = Flatten()(input)
    outputb = Dense(128)(fdf)
    outputc = Dense(64)(outputb)
    outputd = Dense(2, activation='sigmoid')(outputc)
    model = Model(inputs=[input], outputs=outputd)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return(model)


def generate_csv(test_csv_path, X_test, Y_test):
    res = []
    for v in range(0, len(X_test)-10, 10):
        res.append(modela.predict(X_test[v:v+10]))
    clean = []
    for g in res:
        for l in g:
            clean.append(l)
    la = []
    for i, l in enumerate(clean):
        if l[0] > l[1]:
            la.append((Y_test[i], 0))
        else:
            la.append((Y_test[i], 1))
    with open(test_csv_path, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['id', 'prediction'])
        for row in results:
            csv_out.writerow(row)


def validate_dataset(csv_path):
    data_v = read_csv(csv_path)
    dict_data = preprocess(data_v)
    X_valid, Y_valid = convert_to_input(dict_data)
    return(X_valid, Y_valid)


def train(train_csv):
    data = read_csv(train_csv)
    dict_data = preprocess(data)
    X, Y = convert_to_input(dict_data)
    model = model_generate()
    modela.fit(X, Y, epochs=10)
    return(modela)


def test(test_csv):
    data = read_csv(test_csv)
    dict_data = preprocess_test(test_csv)
    X, Y = convert_to_input_test(test_csv)
    return(X, Y)
