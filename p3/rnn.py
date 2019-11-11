import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
from data_loader import fetch_data
from ffnn1fix import make_vocab, make_indices, convert_to_vector_representation

unk = '<UNK>'


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):  # Add relevant parameters
        super(RNN, self).__init__()
        self.RNN = nn.RNN(input_dim, hidden_dim, n_layers)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # begin code
        hidden_state = torch.randn(self.n_layers, self.hidden_dim)
        output, hidden_state = self.RNN(inputs, hidden_state)
        # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
        predicted_vector = self.softmax(output)
        # end code
        # hidden state not needed, here for test
        return predicted_vector


def main(epochs):  # Add relevant parameters
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    train_data, valid_data = fetch_data()
    # convert to one hot encoding
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("Fetched and indexed data")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    print("Vectorized data")
    model = RNN(input_dim=len(vocab), hidden_dim=32)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):  # How will you decide to stop training and why
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)
        for i, document in enumerate(range(N)):
            input_vector, gold_label = train_data[i]
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            if predicted_label == gold_label:
                correct += 1
            updated_loss = model.compute_Loss(
                predicted_vector.view(1, -1), torch.tensor([gold_label]))
            loss += updated_loss
            loss.backward()
            optimizer.step()
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(
                epoch + 1, correct / total))
            print("Training time for this epoch: {}".format(
                time.time() - start_time))
        # Validation loss/acc
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data)
        N = len(valid_data)
        for i, document in enumerate(range(N)):
            input_vector, gold_label = train_data[i]
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            if predicted_label == gold_label:
                correct += 1
            updated_loss = model.compute_Loss(
                predicted_vector.view(1, -1), torch.tensor([gold_label]))
            loss += updated_loss
            loss.backward()
            optimizer.step()
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(
            epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(
            time.time() - start_time))

        # You will need to validate your model. All results for Part 3 should be reported on the validation set.
        # Consider ffnn.py; making changes to validation if you find them necessary
