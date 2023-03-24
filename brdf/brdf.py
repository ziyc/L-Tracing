import torch
import torch.nn as nn
import deepdish as dd

from brdf.latentcode import LatentCode
from models.base import MLP, get_embedder

class Model(nn.Module):
    def __init__(self, config, debug=False):
        super().__init__()
        self.config = config
        # super().__init__(config, debug=debug)
        self.mlp_chunk = self.config.getint('DEFAULT', 'mlp_chunk')
        
        # Embedders
        pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
        n_freqs = self.config.getint('DEFAULT', 'n_freqs')
        if not pos_enc:
            n_freqs=-1
        self.embedder, self.embed_out = get_embedder(n_freqs)
        
        # Network components
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        self.net = MLP(D=mlp_depth,
                       W=mlp_width,
                       skips=[mlp_skip_at+1],
                       embed_N_freqs=-1,
                       W_geo_feat=0,
                       use_geo_feature=False,
                       weight_norm=False,
                       input_chanel=18,
                       final_out_dim=1,
                       out_acti='softplus')
    
        
        normalize_z = self.config.getboolean('DEFAULT', 'normalize_z')
        self.latent_code = LatentCode(normalize=normalize_z)
        
        self.load_para()

    def load_para(self):
        mlp_name = 'brdf'

        brdf_path = './brdf/brdf_ckpt-50.pth'
        para_brdf = dd.io.load(brdf_path)
        
        choice = [self.net.layers[i] for i in range(5)]

        mlp_layers_weight_name = ['%s_mlp_layer%d'%(mlp_name, i) for i in range(4)]
        mlp_layers_weight_name.append(mlp_name + '_out_layer0')
        mlp_layers_bias_name = ['%s_mlp_layer%d_bias'%(mlp_name, i) for i in range(4)]
        mlp_layers_bias_name.append(mlp_name + '_out_layer0_bias')
        
        with torch.no_grad():
            for idx, (weight, bias) in enumerate(zip(mlp_layers_weight_name, mlp_layers_bias_name)):
                layer = choice[idx]
                layer.weight.copy_(torch.from_numpy(para_brdf[weight]).transpose(1,0))
                layer.bias.copy_(torch.from_numpy(para_brdf[bias]))
                
    def forward(self, input):
        return self.net(input)