# Author ~ Naafi Dasana IBRAHIM

''' The following is a from scratch implementation of functions
 and classes to help with building language modelling
 and neural machine translation tools with pytorch.
 This is intended to work with both custom implementations of
 RNN/Transformer nerual network classes or those prebuilt into pytorch.
'''

import collections
import re

import torch
import torch.nn as nn
import math


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens == None:
            tokens = []
        if reserved_tokens == None:
            reserved_tokens = []

        # Sort according to frequency
        counter = count_corpus(tokens)
        self._token_freqs = sorted(
            counter.items(), key=lambda x: x[1], reverse=True)

        # Index for unknown '<unk>' token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx,
                             token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0    # Index for the unknown '<unk>' token

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """Count token frequencies."""
    # Tokens is 1-Dimensional or 2-Dimensional list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists (2D) into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def tokenize(lines, tokenize_by="word"):
    """Split text lines into word or character tokens."""
    if tokenize_by == "word":
        return [line.split() for line in lines]
    elif tokenize_by == "char":
        return [list(line) for line in lines]
    else:
        print(f"ERROR: unknown token type: {tokenize_by}")


def read_dataset(filepath):
    """Read a dataset from the file path specified."""
    with open(filepath, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
    # Remove all special characters that do not form part of the vocabulary
    # and convert all text into lower case.
    return [re.sub('^[A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus(lines, max_tokens=-1):
    """Return token indices and the vocabulary of the dataset."""
    tokens = tokenize(lines)
    vocab = Vocab(tokens)

    # Given the possibility that not all lines in the file is a sentence or paragraph,
    # flatten all text lines into a single list.
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def read_nmt_dataset(filepath):
    """Load the Dagbani-English dataset."""
    with open(filepath, 'r+', encoding='utf-8') as f:
        return f.read()


def tokenize_nmt(text, num_examples=None):
    """Tokenize the Dagbani-English dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def preprocess(text):
    """Preprocess the dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space to space and convert uppercase letters into lower case
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    # Insert space between words and punctuation marks.
    out = [' ' + char if i >
           0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]     # Trucnate
    return line + [padding_token] * (num_steps - len(line))


def build_nmt_array(lines, vocab, num_steps):
    """Transform text sequences into minibatches."""
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = torch.tensor(
        [truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=False):
    """Construct a Pytorch data iterator."""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_nmt_data(filepath, batch_size, num_steps, num_examples=None):
    """Return iterator and vocabulary of the machine translation dataset."""
    text = preprocess(read_nmt_dataset(filepath))
    source, target = tokenize_nmt(text, num_examples)
    src_vocab, trg_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>']), Vocab(
        target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_nmt_array(source, src_vocab, num_steps)
    trg_array, trg_valid_len = build_nmt_array(target, trg_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, trg_array, trg_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, trg_vocab


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[
        None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def try_gpu():
    """Use GPU if CUDA is available."""
    return (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def gradient_clipping(net, theta):
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k+1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i+n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
