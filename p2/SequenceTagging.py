import csv
import numpy as np
import random
import ast
import math
import nltk
import nltk.classify.util
import nltk.metrics
from nltk.classify import MaxentClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn.model_selection import cross_validate
from nltk.classify import MaxentClassifier


def calculate_transition_probs(labels):
    """
    dict of bigram sequences to their counts
    """
    counts = {}
    for label_set in labels:
        for i in range(len(label_set)):
            if i < 1:
                prev_label = "<s>"
                counts["<s>"] = counts.get("<s>", 0) + 1

            else:
                prev_label = str(label_set[i-1])
            curr_label = str(label_set[i])

            counts[curr_label] = counts.get(curr_label, 0) + 1
            counts[prev_label +
                   curr_label] = counts.get(prev_label + curr_label, 0) + 1
    return counts


def hmm_counts(words, labels):
    """

    """
    counts = {"<UNK>": 0}
    x = 0
    for x in range(len(labels)):
        for i in range(len(labels[x])):
            curr_word = words[x][i]
            curr_label = str(labels[x][i])
            if(not curr_word+curr_label in counts.keys()):
                counts["<UNK>"] += 1
            counts[curr_word +
                   curr_label] = counts.get(curr_word + curr_label, 0) + 1
    return counts


def return_featureset(words, pos_seq, i):
    feature_dict = {}
    feature_dict["POS"] = pos_seq[i]
    feature_dict["word"] = words[i]
    if i >= 2:
        feature_dict["2before"] = words[i-2]
        feature_dict["1before"] = words[i-1]
    else:
        feature_dict["2before"] = None
        feature_dict["1before"] = None
    if i < len(words)-2:
        feature_dict["1after"] = words[i+1]
        feature_dict["2after"] = words[i+2]
    else:
        feature_dict["1after"] = None
        feature_dict["2after"] = None

    return feature_dict


def create_classifier():
    with open('./data_release/train.csv', encoding='latin-1') as f:
        feature_set = []
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            label_seq = ast.literal_eval(line[2])
            words = line[0].split()
            pos_seq = ast.literal_eval(line[1])
            for i in range(len(words)):
                feature_dict = return_featureset(words, pos_seq, i)
                feature_set.append((feature_dict, label_seq[i]))

    classifier = nltk.MaxentClassifier.train(feature_set, max_iter=2)
    return classifier


def viterbi_segment(text, pos_seq, counts, classifier, hmmcounts, isHMM):
    n = len(text)
    words = [''] + list(text)

    SCORE = [0]*2
    for i in range(2):
        SCORE[i] = [0] * n
    BPTR = [0]*2
    for i in range(2):
        BPTR[i] = [0] * n

    word0 = "<UNK>"
    word1 = "<UNK>"
    if(text[0]+"0" in hmmcounts.keys()):
        word0 = text[0]+"0"
    elif(text[0]+"1" in hmmcounts.keys()):
        word1 = text[0]+"1"

    feature_dict = return_featureset(words, pos_seq, 0)
    if(isHMM):
        SCORE[0][0] = (counts["<s>0"]/counts["<s>"]) * \
            (hmmcounts[word0]/counts["0"])
        SCORE[1][0] = (counts["<s>0"]/counts["<s>"]) * \
            (hmmcounts[word1]/counts["1"])
    else:
        SCORE[0][0] = (counts["<s>0"]/counts["<s>"]) * \
            classifier.prob_classify(featureset=feature_dict).prob(0)
        SCORE[1][0] = (counts["<s>1"]/counts["<s>"]) * \
            classifier.prob_classify(featureset=feature_dict).prob(1)

    probs = classifier.prob_classify(featureset=feature_dict)
    # SCORE[0][0] =  classifier.prob_classify(featureset=feature_dict).prob(0)
    # SCORE[1][0] =  classifier.prob_classify(featureset=feature_dict).prob(1)
    BPTR[0][0] = None
    BPTR[1][0] = None

    c = 0

    for t in range(1, n):
        if(not isHMM):
            if classifier.prob_classify(feature_dict).prob(0) < classifier.prob_classify(feature_dict).prob(1)+.5:
                # print("HERE")
                # print(classifier.prob_classify(feature_dict).prob(0))
                # print(classifier.prob_classify(feature_dict).prob(1))
                c += 1
            feature_dict = return_featureset(words, pos_seq, t)

            if SCORE[0][t-1]*(counts["00"]/counts["0"]) >= SCORE[1][t-1] * (counts["10"]/counts["1"]):
                # SCORE[0][t] = SCORE[0][t-1]* (counts["00"]/counts["0"]) * classifier.prob_classify(feature_dict).prob(0)
                SCORE[0][t] = SCORE[0][t-1] * \
                    classifier.prob_classify(feature_dict).prob(0)
                BPTR[0][t] = 0
            else:
                # SCORE[0][t] = SCORE[1][t-1]* (counts["10"]/counts["1"]) * classifier.prob_classify(feature_dict).prob(0)
                SCORE[0][t] = SCORE[1][t-1] * \
                    classifier.prob_classify(feature_dict).prob(0)
                BPTR[0][t] = 1

            if SCORE[0][t-1]*(counts["01"]/counts["0"]) >= SCORE[1][t-1] * (counts["11"]+1000/counts["1"]):
                # SCORE[1][t] = SCORE[0][t-1]*(counts["01"]/counts["0"]) * classifier.prob_classify(feature_dict).prob(1)
                SCORE[1][t] = SCORE[0][t-1] * \
                    (classifier.prob_classify(feature_dict).prob(1)+.5)
                BPTR[1][t] = 0
            else:
                # SCORE[1][t] = SCORE[1][t-1]*(counts["11"]/counts["1"]) * classifier.prob_classify(feature_dict).prob(1)
                SCORE[1][t] = SCORE[1][t-1] * \
                    (classifier.prob_classify(feature_dict).prob(1)+.5)
                BPTR[1][t] = 1
        else:
            word0 = "<UNK>"
            word1 = "<UNK>"
            if(text[t]+"0" in hmmcounts.keys()):
                word0 = text[t]+"0"
            elif(text[t]+"1" in hmmcounts.keys()):
                word1 = text[t]+"1"

            if SCORE[0][t-1]*(counts["00"]/counts["0"]) >= SCORE[1][t-1] * (counts["10"]/counts["1"]):
                SCORE[0][t] = math.exp(math.log(SCORE[0][t-1] +
                                                (counts["00"]/counts["0"]) + (hmmcounts[word0]/counts["0"])))
                BPTR[0][t] = 0
            else:
                SCORE[0][t] = math.exp(math.log(SCORE[1][t-1] *
                                                (counts["10"]/counts["1"]) * (hmmcounts[word0]/counts["0"])))
                BPTR[0][t] = 1

            if SCORE[0][t-1]*(counts["01"]/counts["0"]) >= SCORE[1][t-1] + (counts["11"]/counts["1"]):
                SCORE[1][t] = math.exp(math.log(SCORE[0][t-1] +
                                                (counts["01"]/counts["0"]) + (hmmcounts[word1]/counts["1"])))
                BPTR[1][t] = 0
            else:
                SCORE[1][t] = math.exp(math.log(SCORE[1][t-1] *
                                                (counts["11"]/counts["1"]) * (hmmcounts[word1]/counts["1"])))
                BPTR[1][t] = 1

    sequence = []
    counter = n-1

    if SCORE[0][n-1] > SCORE[1][n-1]:
        sequence.append(0)
        a = BPTR[0][counter]
    else:
        sequence.append(1)
        a = BPTR[1][counter]

    while a is not None:
        counter -= 1
        sequence.append(a)
        a = BPTR[a][counter]
    # print(sequence[::-1])
    return sequence[::-1]


def predict_classes():
    total_labels = []
    total_words = []
    with open('./data_release/train.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            word_seq = line[0].split()
            label_seq = ast.literal_eval(line[2])
            total_words.append(word_seq)
            total_labels.append(label_seq)

    counts = calculate_transition_probs(total_labels)
    hmmC = hmm_counts(total_words, total_labels)
    classifier = create_classifier()

    out_file = open("output.csv", "a")


    with open('./data_release/val.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        i=0
        for line in lines:
            words = line[0].split()
            pos_seq = ast.literal_eval(line[1])
            curr_sequence = viterbi_segment(words, pos_seq, counts, classifier, hmmC, True)
            for element in curr_sequence:
                out_file.write(str(i) + "," + str(element) + "\n")
                i+=1
            # print(curr_sequence)


predict_classes()
