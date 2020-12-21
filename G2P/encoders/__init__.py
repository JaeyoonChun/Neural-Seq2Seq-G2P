from G2P.encoders.TransformerEncoder import TransformerEncoder
from G2P.encoders.LSTMEncoder import LSTMEncoder

Encoders_list ={
    "LSTM": LSTMEncoder,
    "Transformer": TransformerEncoder
}

__all__ = ["TransformerEncoder", "LSTMEncoder", "Encoders_list"]