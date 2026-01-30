import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from model.layer import AttentionLayer, FullAttention
from model.FreMo import FreMo
import sys
import argparse
from torchinfo import summary

def init_linear(module: nn.Linear):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

class TemporalEncoder(nn.Module):
    """
        Input:
            h: [b,d,n,m,t]
        Output:
            h_tem_enc: [b,d,n,m,t]
    """
    def __init__(self, num_modes, d, layers, kernel_size, dilation):
        super().__init__()
        self.causal_pad_len = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv2d(in_channels = d, 
                                     out_channels = num_modes * d, 
                                     kernel_size = (num_modes, kernel_size),
                                     dilation=(1, dilation),
                                     padding=0
                                    )
        self.gate_conv = nn.Conv2d(in_channels = d, 
                                   out_channels = num_modes * d, 
                                   kernel_size = (num_modes, kernel_size),
                                   dilation=(1, dilation),
                                   padding=0
                                  )
        self.residual = nn.Conv2d(in_channels = d, out_channels = d, kernel_size=(1,1))

    def forward(self, h):
        # h: [b,d,n,m,t]
        b, d, n, m, t  = h.shape        
        h =h.permute(0, 2, 1, 3, 4).reshape(b*n, d, m, t) 
        # causal padding (left padding only)
        h_pad = F.pad(h, (self.causal_pad_len, 0, 0, 0))
        # filter
        filter_out = self.filter_conv(h_pad)
        filter_act = torch.tanh(filter_out).reshape(b*n, d, m, t)
        # gate
        gate_out = self.gate_conv(h_pad)
        gate_act = torch.sigmoid(gate_out).reshape(b*n, d, m, t)
        h_res = self.residual(filter_act * gate_act)
        h_tem_enc = h_res.reshape(b, n, d, m, t).permute(0, 2, 1, 3, 4)
        return h_tem_enc
    
    
class SpatialEncoder(nn.Module):
    """
        Z <- Attn(Z, nodes)
        nodes <- Attn(nodes, Z)

        Input:
            h: [b,d,n,m,t]
        Output:
            h_spa_enc: [b,d,n,m,t]
    """
    def __init__(self, d, latents, dropout=0.1, n_heads=1):
        super(SpatialEncoder, self).__init__()
        self.latents = nn.Parameter(torch.randn(latents, d))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)

        self.cross_attn1 = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
            d_model=d, n_heads=n_heads
        )
        self.cross_attn2 = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
            d_model=d, n_heads=n_heads
        )

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*d), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout),
            nn.Linear(4*d, d)
        )
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                init_linear(m)

    def forward(self, h):
        b, d, n, m, t  = h.shape
        h_m = h.permute(0, 3, 4, 2, 1).reshape(b*m*t, n, d)
        Z = self.latents.repeat(b*m*t,1,1)

        # Cross-Attn 1: Z <- Attn(Z, nodes)
        z1, _ = self.cross_attn1(Z, h_m, h_m, attn_mask=None)
        Z = self.ln1(Z + z1)
        
        # Cross-Attn 2: nodes <- Attn(nodes, Z)
        n1, _ = self.cross_attn2(h_m, Z, Z, attn_mask=None)
        nodes = self.ln2(h_m + n1)

        # FFN
        h_spa_enc = nodes + self.ffn(nodes) 
        h_spa_enc = h_spa_enc.reshape(b,m,t,n,d).permute(0,4,3,1,2) 
        return h_spa_enc


class MultiModalSTEncoder(nn.Module):
    """
        Input: 
            h: [b,d,n,m,t]
        Output: 
            h_enc: [b,d,n,m,t]
    """
    def __init__(self, num_modes, d, layers, latents, kernel_size=3):
        super(MultiModalSTEncoder, self).__init__()
        dilations = [2**i for i in range(layers)]
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'temporal': TemporalEncoder(num_modes, d, layers, kernel_size, dilations[i]),
                'spatial': SpatialEncoder(d, latents)
            }) for i in range(layers)
        ])

    def forward(self, h):
        for layer in self.layers:
            h_in = h
            # Temporal
            h_tem_enc = layer['temporal'](h)
            h = h_in + h_tem_enc
            # Spatial
            h_spa_enc = layer['spatial'](h)
            h = h_in + h_spa_enc
        return h
    

class Predictor(nn.Module):
    """
        GLU-based temporal selection
        input:
            h: [b,d,n,m,t]
        output:
            pred: [b,pred_len,n,m,1]
    """
    def __init__(self, d, pred_len):
        super().__init__()

        self.temporal_glu = nn.Conv3d(
            in_channels=d,
            out_channels=d * 2,
            kernel_size=(1, 1, 16)
        )

        self.proj = nn.Conv3d(
            in_channels=d,
            out_channels=pred_len,
            kernel_size=(1, 1, 1)
        )

    def forward(self, h):
        h_glu = self.temporal_glu(h)
        v, g = h_glu.chunk(2, dim=1)
        h = v * torch.sigmoid(g)
        pred = self.proj(h)
        return pred
    
    
class model(nn.Module):
    """
    Parameter:
        d: int = 64                               # dimention of hidden channel
        seq_len: int = 16                         # input length
        pred_len: int = 3                         # horizon
        num_nodes: int = 98                       # nodes
        num_modes: int = 4                        # modalities
        L: int = 64                               # inducing Points (latent)
        layers: int = 4                           # TCN layer
        kernel: int = 3                           # TCN kernel
    """
    def __init__(self, seq_len, num_nodes, num_modes, pred_len, d, latents, layers, in_dim=1, kernel=3):
        super(model, self).__init__()
        # Linear Projection
        self.init_proj = nn.Sequential(
            nn.Conv3d(in_channels = in_dim, out_channels = int(d//2), kernel_size = (1,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channels = int(d//2), out_channels = d, kernel_size = (1,1,1)),
            nn.ReLU()
        )  
        
        # General Encoder
        self.encoder = MultiModalSTEncoder(num_modes, d, layers, latents, kernel)
        
        # Frequency-Domain Multi-Modality Transportation Modeling
        self.fre_encoder = FreMo(seq_len, num_nodes, num_modes, d, latents)
        
        # Predictor
        self.predictor = Predictor(d, pred_len)
        
    def forward(self, input):
        x = input.permute(0, 4, 2, 3, 1)
        h_init = self.init_proj(x) 
        # General Encoder
        h_enc = self.encoder(h_init)
        # Frequency-Domain Multi-Modality Transportation Modeling
        h_fre = self.fre_encoder(h_enc) 
        # Predictor
        pred = self.predictor(h_fre)
        return pred        
                
