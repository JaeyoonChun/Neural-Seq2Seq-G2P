from G2P.decoders.TransformerDecoder import TransformerDecoder
from G2P.decoders.LSTMDecoder import LSTMDecoder

Decoders_list ={
    "LSTM": LSTMDecoder,
    "Transformer": TransformerDecoder
}

__all__ = ["TransformerDecoder", "LSTMDecoder", "Decoders_list"]