from G2P.encoders import Encoders_list
from G2P.decoders import Decoders_list
from G2P.models.model import G2P

def build_encoder(model_type, model_args, fields, device):
    return Encoders_list[model_type].from_args(model_args, fields, device)

def build_decoder(model_type, model_args, fields, device):
    return Decoders_list[model_type].from_args(model_args, fields, device)

def build_model(model_args, opt, fields, device):
    encoder = build_encoder(opt.model_type, model_args, fields, device)
    decoder = build_decoder(opt.model_type, model_args, fields, device)
    
    return G2P(encoder=encoder, decoder=decoder)
