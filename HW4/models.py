# models.py

import numpy as np
import random
import time
import collections
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable as Var


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index
        
    def parse_context(self, context):
        curr_state = (torch.from_numpy(np.zeros(self.model_dec.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.model_dec.hidden_size)).unsqueeze(0).unsqueeze(1).float())

        context = [self.vocab_index.index_of(c) for c in context]
        context = np.asarray(context)

        for i in range(0, len(context) - 1):
            input_th = torch.from_numpy(np.asarray(context[i]))
            embedded = self.model_emb.forward(input_th)
            _, curr_state = self.model_dec.forward(embedded, curr_state)
        input_th = torch.from_numpy(np.asarray(context[-1]))
        
        return input_th, curr_state

    def get_next_char_log_probs(self, context):
        input_th, curr_state = self.parse_context(context)

        embedded = self.model_emb.forward(input_th)
        log_probs, _ = self.model_dec.forward(embedded, curr_state)
        log_probs = log_probs.squeeze()
        
        return log_probs.detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        input_th, curr_state = self.parse_context(context)
        input = [self.vocab_index.index_of(c) for c in next_chars]
        input = np.asarray(input)
        
        seq_prob = 0
        for i in range(0, len(input)):
            embedded = self.model_emb.forward(input_th)
            log_probs, curr_state = self.model_dec.forward(embedded, curr_state)
            seq_prob += log_probs.squeeze()[input[i]]
            input_th = torch.from_numpy(np.asarray(input[i]))
        
        return seq_prob.item()


class TransformerLanguageModel(LanguageModel):
    def __init__(self, model_dec, vocab_index):
        self.model_dec = model_dec
        self.vocab_index = vocab_index
        
    def parse_context(self, context):
        context = [self.vocab_index.index_of(c) for c in context]
        context = np.asarray(context)

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


#####################
#     RNN Decoder   #
#####################


class RNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_dict_size, dropout, rnn_type='lstm'):
        super(RNNDecoder, self).__init__()
        self.n_layers = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_input_size = input_size
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.rnn = nn.GRU(self.cell_input_size,
                              hidden_size, dropout=dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.cell_input_size,
                               hidden_size, num_layers=1, dropout=dropout)
        else:
            raise NotImplementedError
        # output should be batch x output_dict_size
        self.output_layer = nn.Linear(hidden_size, output_dict_size)
        # print(f"Out dict size {output_dict_size}")
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.init_weight()

    def init_weight(self):
        if self.rnn_type == 'lstm':
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)

            nn.init.constant_(self.rnn.bias_hh_l0, 0)
            nn.init.constant_(self.rnn.bias_ih_l0, 0)
        elif self.rnn_type == 'gru':
            nn.init.xavier_uniform_(self.rnn.weight.data, gain=1)

    def forward(self, embedded_input, state):
        output, state = self.rnn(embedded_input.unsqueeze(0).unsqueeze(0), state)
        output = self.output_layer(output)
        output = self.log_softmax_layer(output.squeeze(0))
        return output, state


#####################
#Transformer Decoder#
#####################


class TransformerDecoder(nn.Module):
    # classification task
    def __init__(self, d, h, depth, max_len, vocab_size, num_classes):
        super(TransformerDecoder, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_len, d)

        trans_blocks = []
        for i in range(depth):
            trans_blocks.append(TransformerBlock(d, h, mask=True))
        self.trans_blocks = nn.Sequential(*trans_blocks)

        self.out_layer = nn.Linear(d, num_classes)

    def forward(self, x):
        # x = (b, t)

        # step 1: get token and position embeddings
        tokens = self.token_emb(x) # (b, t, e)
        b, t, e = tokens.size()
        
        pos = torch.arange(t)
        pos = self.pos_emb(x)[None, :, :].expand(b, t, e)
        # (t) => (t, e) => (b, t, e)
        
        # step 2: pass them through Transformer blocks
        x = self.trans_blocks(tokens + pos)
        # (b, t, e)

        # step 3: pass the outputlayer
        x = self.out_layer(x.mean(dim=1))
        # (b, t, e) => (b, e) => (b, num_classes)

        # step 4: pass the log_softmax layer
        return F.log_softmax(x, dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, d, h, mask):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(d, h, mask=mask)

        self.norm_1 = nn.LayerNorm(d)
        self.norm_2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, d)
        )

    def forward(self, x):

        # step 1: self-attention
        attended = self.attention(x)

        # step 2: residual + layer norm
        x = self.norm_1(attended + x)

        # step 3: FFN/MLP
        feedforward = self.ff(x)

        # step 4: residual + layer norm
        return self.norm_2(feedforward + x)


class SelfAttention(nn.Module):
    def __init__(self, d, h=8, mask=False):
        super(SelfAttention, self).__init__()
        # d: dimension
        # heads: number of heads
        self.d, self.h = d, h
        self.mask = mask
        self.linear_key = nn.Linear(d, d * h, bias=False)
        self.linear_query = nn.Linear(d, d * h, bias=False)
        self.linear_value = nn.Linear(d, d * h, bias=False)
        self.linear_unify = nn.Linear(d * h, d)

    def forward(self, x):
        b, t, e = x.size()
        h = self.h
        
        # step 1: transform x to key/query/value
        keys = self.linear_key(x).view(b, t, h, e)
        queries = self.linear_query(x).view(b, t, h, e)
        values = self.linear_value(x).view(b, t, h, e)
        # (b, t, e) => (b, t, h*e) => (b, t, h, e)
        
        keys = keys.transpose(1, 2).contiguous().view(b*h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b*h, t, e)
        values = values.transpose(1, 2).contiguous().view(b*h, t, e)
        # (b, t, h, e) => (b*h, t, e)

        # step 2: scaled dot product between key and query to get attention
        dot = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(e)
        # (b*h, t, e) * (b*h, e, t) => (b*h, t, t)

        # step 3: casual masking (you may use torch.triu_indices)
        if self.mask:
            mask = torch.triu_indices(t, t, offset=1)
            dot[:, mask[0], mask[1]] = float('-inf')
            # (b*h, t, t)

        # step 4: softmax over attention
        dot = F.softmax(dot, dim=2)
        # (b*h, t, t)
        
        # step 5: multiply attention with value
        out = torch.bmm(dot, values).view(b, h, t, e)
        # (b*h, t, t) * (b*h, t, e) => (b*h, t, e) => (b, h, t, e)
        
        # step 6: another linear layer for output
        out = out.transpose(1, 2).contiguous().view(b, t, h*e)
        return self.linear_unify(out)
        # (b, h, t, e) => (b, t, h*e) => (b, t, e)
