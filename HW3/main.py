import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
from typing import List


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='main.py')

    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor',
                        default=False, action='store_true', help='run the nearest neighbor model')
    parser.add_argument('--attention', dest='attention',
                        default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str,
                        default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str,
                        default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str,
                        default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str,
                        default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo',
                        help='domain (geo for geoquery)')

    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.1)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int,
                        default=65, help='output length limit of the decoder')
    parser.add_argument('--embed_size', type=int, default=200, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes
    args = parser.parse_args()
    return args


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """

    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap / \
                    float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self, enc_emb, encoder, decoder, output_indexer, output_max_len, args):
        self.enc_emb = enc_emb
        self.encoder = encoder
        self.decoder = decoder
        self.output_indexer = output_indexer
        self.output_max_len = output_max_len
        self.args = args

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        result = []
        
        # Loop through shuffled examples
        for example in test_data:
            input = torch.tensor([example.x_indexed])
            i_len = torch.tensor([len(example.x_indexed)])

            # Encode input
            encoder_outputs, _, hidden = encode_input_for_decoder(input, i_len, self.enc_emb, self.encoder)

            # Generate output until <EOS> token or max output length is reached
            words = []
            decoder_input = torch.tensor([[self.output_indexer.index_of("<SOS>")]])
            decoder_context = torch.zeros(1, args.hidden_size)
            for _ in range(self.output_max_len):
                if args.attention:
                    decoder_output, hidden, decoder_context = self.decoder(decoder_input, hidden, decoder_context, encoder_outputs)
                else:
                    decoder_output, hidden = self.decoder(decoder_input, hidden)
                choice = torch.argmax(decoder_output).item()
                word = self.output_indexer.get_object(choice)
                if word == "<EOS>": break
                words.append(word)
                decoder_input = torch.tensor([[choice]])
            result.append([Derivation(example, 1.0, words)])
        return result


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int, reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])


def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb: EmbeddingLayer, model_enc: RNNEncoder):
    """
    Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
    inp_lens_tensor lengths.
    YOU DO NOT NEED TO USE THIS FUNCTION. It's merely meant to illustrate the usage of EmbeddingLayer and RNNEncoder
    as they're given to you, as well as show what kinds of inputs/outputs you need from your encoding phase.
    :param x_tensor: [batch size, sent len] tensor of input token indices
    :param inp_lens_tensor: [batch size] vector containing the length of each sentence in the batch
    :param model_input_emb: EmbeddingLayer
    :param model_enc: RNNEncoder
    :return: the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting which words
    are real and which ones are pad tokens), and the encoder final states (h and c tuple)
    E.g., calling this with x_tensor (0 is pad token):
    [[12, 25, 0, 0],
    [1, 2, 3, 0],
    [2, 0, 0, 0]]
    inp_lens = [2, 3, 1]
    will return outputs with the following shape:
    enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
    enc_final_states = 3 x dim
    """
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data: List[Example], test_data: List[Example], input_indexer, output_indexer, args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param test_data:
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_len = np.array([len(ex.x_indexed) for ex in train_data])
    input_max_len = np.max(input_len)
    all_train_input_data = make_padded_input_tensor( train_data, input_indexer, input_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, reverse_input=False)
    
    output_len = np.array([len(ex.y_indexed) for ex in train_data])
    output_max_len = np.max(output_len)
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % output_max_len)
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Initialize model
    enc_emb = EmbeddingLayer(args.embed_size, len(input_indexer), 0)
    encoder = RNNEncoder(args.embed_size, args.hidden_size, False)
    if args.attention:
        decoder = RNNAttnDecoder(args.embed_size, args.hidden_size, len(output_indexer))
    else:
        decoder = RNNDecoder(args.embed_size, args.hidden_size, len(output_indexer))

    # Initialize optimizer
    enc_emb_optimizer = optim.Adam(enc_emb.parameters())
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    criterion = nn.NLLLoss()

    # Loop through epochs
    data = list(zip(all_train_input_data, input_len, all_train_output_data, output_len))
    for epoch in range(args.epochs):
        start = time.time()
        total_loss = 0
        random.shuffle(data)

        # Loop through shuffled examples
        for input, i_len, target, t_len in data:
            input = torch.from_numpy(input.reshape(1, input_max_len)) # [1 x 19]
            i_len = torch.tensor([i_len]) # [1]
            target = torch.from_numpy(target) # [65]
        
            # Zero gradients
            enc_emb_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Encode input
            # encoder_output:   [input_length x 1 x 200]
            # context_mask:     [1 x input_length]
            # hidden:           ([1 x 1 x 200], [1 x 1 x 200])
            encoder_outputs, _, hidden = encode_input_for_decoder(input, i_len, enc_emb, encoder)

            # Accumulating error w/ teacher forcing 
            # decoder_input:    [1 x 1]
            # decoder_context:  [1 x 200]
            # decoder_outputs:   [1 x 153]
            loss = 0
            decoder_input = torch.tensor([[output_indexer.index_of("<SOS>")]])
            decoder_context = torch.zeros(1, args.hidden_size)
            for di in range(t_len):
                if args.attention:
                    decoder_output, hidden, decoder_context = decoder(decoder_input, hidden, decoder_context, encoder_outputs)
                else:
                    decoder_output, hidden = decoder(decoder_input, hidden)
                loss += criterion(decoder_output, target[di:di+1])
                decoder_input = torch.tensor([[target[di]]])

            # Optimizing parameters
            loss.backward()
            enc_emb_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}\tTime: {round(time.time() - start, 2)}\tLoss: {total_loss}")
    return Seq2SeqSemanticParser(enc_emb, encoder, decoder, output_indexer, output_max_len, args)


def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    java_crashes = False
    if java_crashes:
        selected_derivs = [derivs[0] for derivs in pred_derivations]
        denotation_correct = [False for derivs in pred_derivations]
    else:
        selected_derivs, denotation_correct = e.compare_answers(
            [ex.y for ex in test_data], pred_derivations, quiet=True)
    print_evaluation_results(test_data, selected_derivs,
                             denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" +
                          " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        evaluate(dev_data_indexed, decoder)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")
