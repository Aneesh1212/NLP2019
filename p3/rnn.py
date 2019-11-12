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
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super(RNN, self).__init__()
        self.RNN = nn.RNN(input_dim, hidden_dim, n_layers)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.fcn = nn.Linear(32, 5)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs, hidden_state=None):
        output, hidden_state = self.RNN(inputs, hidden_state)
        linear_output = self.fcn(hidden_state.view(1, -1))
        output = self.softmax(linear_output)
        return output, hidden_state


def main(epochs, speed_up=1):
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    minibatch_size = 16
    train_data, valid_data = fetch_data()
    # convert to one hot encoding
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("Fetched and indexed data")
    print("Vectorized data")
    model = RNN(input_dim=len(vocab), hidden_dim=32)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = 0.00
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)
        minibatch_size = 100
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss_g = 0.00
            for example_index in range(minibatch_size):
                total += 1
                input_vector = train_data[minibatch_index *
                                          minibatch_size + example_index]

                gold_label = input_vector[1]
                input_vector = convert_to_vector_representation_rnn(
                    input_vector, word2index)
                # force teacher learning w/ 50% success
                if random.random() < 0.5:
                    force = True
                else:
                    force = False
                predicted_vector, hidden = model(input_vector, None)
                predicted_probabilities = torch.exp(predicted_vector)
                predicted_label = torch.argmax(predicted_probabilities)
                if predicted_label == gold_label:
                    correct += 1
                updated_loss = model.compute_Loss(
                    predicted_vector.view(1, -1), torch.tensor([gold_label]))
                loss_g += updated_loss
            loss = loss_g / minibatch_size
            loss.backward(retain_graph=True)
            optimizer.step()
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(
                epoch + 1, correct / total))
            print("Training time for this epoch: {}".format(
                time.time() - start_time))
            print("Loss for this epoch: {}".format(
                loss))

            # Validation loss/acc
            loss = 0.00
            correct = 0
            total = 0
            start_time = time.time()
            print("Validation started for epoch {}".format(epoch + 1))
            random.shuffle(valid_data)
            N = int(len(valid_data) / speed_up)
            print(N)
            for i, document in enumerate(range(N)):
                total += 1
                input_vector = valid_data[i]
                gold_label = input_vector[1]
                input_vector = convert_to_vector_representation_rnn(
                    input_vector, word2index)
                predicted_vector, hidden = model(input_vector, None)
                predicted_label = torch.argmax(predicted_vector)
                if predicted_label == gold_label:
                    correct += 1
                updated_loss = model.compute_Loss(
                    predicted_vector.view(1, -1), torch.tensor([gold_label]))
                loss += updated_loss
                loss.backward(retain_graph=True)
                optimizer.step()
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(
                epoch + 1, correct / total))
            print("Validation time for this epoch: {}".format(
                time.time() - start_time))

    return(model)
