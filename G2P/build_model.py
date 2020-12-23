from G2P.encoders import Encoders_list
from G2P.decoders import Decoders_list
from G2P.models.model import G2P
from torch.nn.init import xavier_normal_

def build_encoder(model_type, model_args, fields, device):
    return Encoders_list[model_type].from_args(model_args, fields, device)

def build_decoder(model_type, model_args, fields, device):
    return Decoders_list[model_type].from_args(model_args, fields, device)

def build_model(model_args, opt, fields, device):
    encoder = build_encoder(opt.model_type, model_args, fields, device)
    decoder = build_decoder(opt.model_type, model_args, fields, device)
    model = G2P(encoder=encoder, decoder=decoder)
    
    if model_args.param_init != 0.0:
        for p in model.parameters():
            p.data.normal_(0, model_args.param_init)
    if model_args.param_init_xavier:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_normal_(p)
    return model
