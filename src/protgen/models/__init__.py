"""All models take token ids (B, T) and return logits (B, T, vocab_size)."""

from .mlp import MLPLM
from .rnn import RNNLM
from .transformer import TransformerLM

MODEL_NAMES = ["mlp", "rnn", "gru", "lstm", "transformer"]


def build_model(name, vocab_size, **kwargs):
    if name == "mlp":
        return MLPLM(vocab_size, **kwargs)
    if name == "transformer":
        return TransformerLM(vocab_size, **kwargs)
    return RNNLM(vocab_size, cell=name, **kwargs)


__all__ = ["MODEL_NAMES", "MLPLM", "RNNLM", "TransformerLM", "build_model"]
