import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from model.layer import AttentionLayer, FullAttention
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
        h_res = self.residual(filter_act * gate_act)   # [b*n, d, m, -1] 
        h_tem_enc = h_res.reshape(b, n, d, m, t).permute(0, 2, 1, 3, 4)  # [b,d,n,m,t]
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
        # h: [b,d,n,m,t]
        b, d, n, m, t  = h.shape
        h_m = h.permute(0, 3, 4, 2, 1).reshape(b*m*t, n, d)   # [b*m*t, n, d]
        Z = self.latents.repeat(b*m*t,1,1)   # [b*m*t, l, d]

        # Cross-Attn 1: Z <- Attn(Z, nodes)
        z1, _ = self.cross_attn1(Z, h_m, h_m, attn_mask=None)
        Z = self.ln1(Z + z1)   # [b*m*t, L, d]

        # Cross-Attn 2: nodes <- Attn(nodes, Z)
        n1, _ = self.cross_attn2(h_m, Z, Z, attn_mask=None)
        nodes = self.ln2(h_m + n1)  # [b*m*t, n, d]

        # FFN
        h_spa_enc = nodes + self.ffn(nodes)  # [b*m*t, n, d]
        h_spa_enc = h_spa_enc.reshape(b,m,t,n,d).permute(0,4,3,1,2)   # [b,d,n,m,t]
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
        # h: [b,d,n,m,t]
        for layer in self.layers:
            h_in = h
            # Temporal
            h_tem_enc = layer['temporal'](h)
            h = h_in + h_tem_enc
            # Spatial
            h_spa_enc = layer['spatial'](h)
            h = h_in + h_spa_enc
        return h
    

class FreMo(nn.Module):
    """
        Args:
            Input h: [b,d,n,m,t]     
        Returns:
            Input out: [b,d,n,m,t]     
        """
    def __init__(self, seq_len, num_nodes, num_modes, d, node_emb_dim=32):
        super(FreMo, self).__init__()
        self.num_modes = num_modes
        self.num_nodes = num_nodes
        self.freq_dim = seq_len // 2 + 1
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_emb_dim))

        input_dim = self.freq_dim + node_emb_dim
        self.weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d),
                nn.ReLU(),
                nn.Linear(d, self.freq_dim),
                nn.Sigmoid() # 输出 [0, 1] 的门控
            ) for _ in range(num_modes)
        ])

        # Modality-shared Scalar
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, h):
        """
        Args:
            h: 输入隐藏状态，形状 [b,d,n,m,t]     
        Returns:
            out: 输出隐藏状态，形状 [b,d,n,m,t]     
        """
        b, d, n, m, t = h.shape
        h_t = h.permute(0, 2, 3, 1, 4)
        # =============================================================
        # FFT over temporal domain
        # =============================================================        
        f_complex = torch.fft.rfft(h_t, dim=-1)
        # Amplitude
        f_amp = torch.abs(f_complex)

        # Node Embedding
        node_emb_expanded = self.node_emb.unsqueeze(0).expand(b, -1, -1)

        f_weighted_list = []
        scores_list = []

        # =============================================================
        # Modality-Specific Frequency Filter (MFF) 
        # =============================================================
        # Per-Mode Processing
        for i in range(m):
            # Spectral signal of modality i
            f_amp_m = f_amp[:, :, i, :, :]
            f_complex_m = f_complex[:, :, i, :, :]
            
            # hidden dimension aggregate
            f_profile = f_amp_m.mean(dim=2)
            
            # Node embedding
            features = torch.cat([f_profile, node_emb_expanded], dim=-1)
            
            # Weight generation
            w = self.weight_generators[i](features)
            
            # Frequency re-weight (Phase Preserving)
            w_expanded = w.unsqueeze(2)  
            f_weighted = f_complex_m * w_expanded
            
            # Synergy score
            score = torch.abs(f_weighted).mean(dim=2)
            scores_list.append(score)

        # =============================================================
        # Frequency-Guided Synergy Augmenter (FSA) 
        # =============================================================
        # Modality softmax
        scores_stack = torch.stack(scores_list, dim=0)
        alpha = F.softmax(scores_stack, dim=0)
        alpha_expanded = alpha.unsqueeze(3)
        
        # Stack filtered features
        f_weighted_stack = torch.stack(f_weighted_list, dim=0)

        # Synergy Consensus
        f_consensus = torch.sum(alpha_expanded * f_weighted_stack, dim=0)
        
        # Feedback & Reconstruction
        f_consensus_expanded = f_consensus.unsqueeze(2) 
        f_out_complex = f_complex + self.gamma * f_consensus_expanded

        # =============================================================
        # Back to temporal domain
        # =============================================================        
        out_perm = torch.fft.irfft(f_out_complex, n=t, dim=-1)
        out = out_perm.permute(0, 3, 1, 2, 4)
        return out

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
        # h: [b,d,n,m,t]
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
        
        # Multi-Modal Spati0-Temporal Encoder
        self.encoder = MultiModalSTEncoder(num_modes, d, layers, latents, kernel)
        
        # Modality-specific Dynamic Frequency Heterogeneity and Synergy Consensus
        self.fre_encoder = FreqAwareModule(seq_len, num_nodes, num_modes, d, latents)
        
        # Predictor
        self.predictor = Predictor(d, pred_len)
        
    def forward(self, input):
        x = input.permute(0, 4, 2, 3, 1)  # [b,t,n,m,1] -> [b,1,n,m,t]
        h_init = self.init_proj(x)        # [b,d,n,m,t]
        # Multi-Modal Spati0-Temporal Encoder
        h_enc = self.encoder(h_init)      # [b,d,n,m,t]
        # Modality-specific Dynamic Frequency Heterogeneity and Synergy Consensus
        h_fre = self.fre_encoder(h_enc)   # [b,d,n,m,t]
        # Predictor
        pred = self.predictor(h_fre)  # [b,pred_len,n,m,1]        
        return pred        
                
