# train.py
import numpy as np
import random
import time
import collections

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable as Var

from models import *


def form_input_output(vocab_index, text, start_idx, chunk_size):
    raw_words = [vocab_index.index_of(text[i]) for i in range(start_idx, start_idx + chunk_size)]
    output = np.asarray(raw_words)
    input = np.asarray([vocab_index.index_of(' ')] + raw_words[:-1])
    return (input, output)

    
def train_rnn_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    emb_dim = 20
    hidden_size = 50
    emb_dropout = 0.0
    rnn_dropout = 0.0
    model_emb = EmbeddingLayer(emb_dim, len(vocab_index), emb_dropout)
    model_dec = RNNDecoder(emb_dim, hidden_size, len(vocab_index), rnn_dropout)
    model_parts = [model_emb, model_dec]
    for model in model_parts:
        model.zero_grad()
        model.train()
    optimizer_parts = [optim.Adam(model.parameters(), lr=1e-3) for model in model_parts]

    burn_in = 0 # ignore this for now
    chunk_len = 20
    batch_starts = [i * (chunk_len - burn_in) for i in range(0, int(len(train_text) / chunk_len))]

    num_epochs = 5

    for t in range(0, num_epochs):
        epoch_start = time.time()
        loss_this_epoch = 0.0
        # Train on chunks of 20 chars with burn in of 5
        random.shuffle(batch_starts)
        num_done = 0
        for batch_idx in batch_starts:
            if num_done % 100 == 0:
                print(repr(num_done))

            (input, output) = form_input_output(vocab_index, train_text, batch_idx, chunk_len)
            loss_fcn = nn.NLLLoss()
            loss = 0

            curr_state = (torch.from_numpy(np.zeros(model_dec.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                          torch.from_numpy(np.zeros(model_dec.hidden_size)).unsqueeze(0).unsqueeze(1).float())

            for i in range(0, len(input)):
                input_th = torch.from_numpy(np.asarray(input[i]))
                output_th = torch.from_numpy(np.asarray(output[i]))
                embedded = model_emb.forward(input_th)
                log_probs, hidden = model_dec.forward(embedded, curr_state)
                log_probs = log_probs.squeeze()
                curr_state = hidden
                y_onehot = torch.from_numpy(np.asarray([0 if j != output[i] else 1 for j in range(0, len(vocab_index))])).float()
                loss += - log_probs.dot(y_onehot)
            loss_this_epoch += loss.item() / len(train_text)
            for model in model_parts:
                model.zero_grad()
            loss.backward()
            [optimizer.step() for optimizer in optimizer_parts]
            num_done += 1

        print(repr(f"Finished epoch {t} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
        decoder = RNNLanguageModel(model_emb, model_dec, vocab_index)
        if t % 10 == 9:
            for model_part in model_parts:
                model_part.eval()
    for model_part in model_parts:
        model_part.eval()
    return decoder



def train_transformer_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an TransformerLanguageModel instance trained on the given data
    """
    emb_dim = 50
    hidden_size = 50
    num_layers = 2
    max_length = 512
    model_dec = TransformerDecoder(emb_dim, hidden_size, num_layers, max_length, len(vocab_index), len(vocab_index))
    model_dec.zero_grad()
    model_dec.train()
    optimizer = optim.Adam(model_dec.parameters(), lr=1e-3)

    burn_in = 0
    chunk_len = 20
    batch_starts = [i * (chunk_len - burn_in) for i in range(0, int(len(train_text) / chunk_len))]

    num_epochs = 5
    
    for t in range(0, num_epochs):
        epoch_start = time.time()
        loss_this_epoch = 0.0
        
        random.shuffle(batch_starts)
        num_done = 0
        for batch_idx in batch_starts:
            if num_done % 100 == 0:
                print(repr(num_done))
                
            (input, output) = form_input_output(vocab_index, train_text, batch_idx, chunk_len)
            loss_fcn = nn.NLLLoss()
            loss = 0
            
            log_probs = model_dec.forward(torch.from_numpy(np.asarray([input])))
            log_probs = log_probs.squeeze()
            y_onehot = torch.from_numpy(np.asarray([0 if j != output[-1] else 1 for j in range(0, len(vocab_index))])).float()
            loss = -log_probs.dot(y_onehot)
            loss_this_epoch += loss.item() / len(train_text)
            model_dec.zero_grad()
            loss.backward()
            optimizer.step()
            num_done += 1

        print(repr(f"Finished epoch {t} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
        decoder = TransformerLanguageModel(model_dec, vocab_index)
        if t % 10 == 9:
            model_dec.eval()
    model_dec.eval()
    return decoder
