import torch
import torch.nn as nn
import utils


class Encoder(nn.Module):
    """The base encoder interface for encoder-decoder architectures."""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax Cross-Entropy Loss with masks."""
    # 'pred' shape: ('batch_size', 'num_steps', 'vocab_size')
    # 'label' shape: ('batch_size', 'num_steps')
    # 'valid_len' shape: ('batch_size',)

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = utils.sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def forward(self):
        raise NotImplementedError
