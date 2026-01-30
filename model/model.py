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
    

class FreqAwareModule(nn.Module):
    def __init__(self, seq_len, num_nodes, num_modes, d, node_emb_dim=32):
        super(FreqAwareModule, self).__init__()
        self.num_modes = num_modes
        self.num_nodes = num_nodes
        self.freq_dim = seq_len // 2 + 1
        
        # 1. 节点嵌入 (Node Embedding)：用于解决空间异质性
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_emb_dim))

        # 2. 模态特异性权重生成器 (Modality-Specific Weight Generators)
        input_dim = self.freq_dim + node_emb_dim
        
        # 为每个模态单独建模
        self.weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d),
                nn.ReLU(),
                nn.Linear(d, self.freq_dim),
                nn.Sigmoid() # 输出 [0, 1] 的门控
            ) for _ in range(num_modes)
        ])
        
        # 3. 协同投影器 (Synergy Projectors)：用于计算 Softmax 注意力分数
#         self.synergy_projectors = nn.ModuleList([
#             nn.Linear(self.freq_dim, self.freq_dim) 
#             for _ in range(num_modes)
#         ])
        
        # 4. 共识残差的缩放参数(对Consensus的听劝程度)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        # Static Modality-Specific Gamma: 因为参数量增加导致难以优化
#         self.gamma = nn.Parameter(torch.zeros(1, 1, num_modes, 1, 1))
        # Dynamic Gating
#         self.gamma_generator = nn.Sequential(
#             nn.Linear(self.freq_dim, d // 4), 
#             nn.ReLU(),
#             nn.Linear(d // 4, 1),
#             nn.Sigmoid()
#         )

    def forward(self, h):
        """
        Args:
            h: 输入隐藏状态，形状 [b,d,n,m,t]     
        Returns:
            out: 输出隐藏状态，形状 [b,d,n,m,t]     
        """
        b, d, n, m, t = h.shape
        h_t = h.permute(0, 2, 3, 1, 4) # [b,d,n,m,t] -> [b, n, m, d, t]
        # =============================================================
        # Step 1: 频域映射 (FFT) -> FFT over time
        # =============================================================        
        f_complex = torch.fft.rfft(h_t, dim=-1) # [b, n, m, d, f]
        f_amp = torch.abs(f_complex) # 取幅值

        # Node Embedding: [n, d] -> [b, n, d]
        node_emb_expanded = self.node_emb.unsqueeze(0).expand(b, -1, -1)

        f_weighted_list = [] # 存储被权重过滤后的各模态特征
        scores_list = []     # 存储注意力分数
        
        # for save
        w_list = []

        # =============================================================
        # Step 2: 逐模态处理 (Per-Mode Processing)
        # =============================================================
        for i in range(m):
            # 取出当前模态的数据
            f_amp_m = f_amp[:, :, i, :, :] # [b, n, d, f]
            f_complex_m = f_complex[:, :, i, :, :]

            # --- A. 节点级频率权重生成 ---
            # 1. 通道汇聚
            f_profile = f_amp_m.mean(dim=2) # [b, n, d, f] -> [b, n, f]
            
            # 2. 注入空间异质性
            features = torch.cat([f_profile, node_emb_expanded], dim=-1) # [b, n, f+d]
            
            # 3. 生成频率权重
            w = self.weight_generators[i](features) # [b, n, f]
            # for save
            w_list.append(w)
            
            # 4. 施加权重 (相位保护): [b, n, d, f] * [b, n, 1, f]
            w_expanded = w.unsqueeze(2)     # [b, n, 1, f]
            f_weighted = f_complex_m * w_expanded  # [b, n, d, f]
            f_weighted_list.append(f_weighted)
            
            # --- B. 计算协同分数 ---
            # 基于过滤后的特征判断质量
#             f_weighted_profile = torch.abs(f_weighted).mean(dim=2)  # [b, n, f]
#             score = self.synergy_projectors[i](f_weighted_profile)  # [b, n, f]
            
            # 直接使用 Profile 作为 Score，移除 Projector:保留下来的能量越大，说明该模态在该频率越重要
            score = torch.abs(f_weighted).mean(dim=2)  # [b, n, f]
            scores_list.append(score)

        # =============================================================
        # Step 3: 频率引导的协同建模 (Synergy Consensus)
        # =============================================================
        # 1. 计算全局模态注意力 (Softmax)
        scores_stack = torch.stack(scores_list, dim=0)  # [m, b, n, f]
        # 在模态维度归一化 -> 得到每个模态在每个频率点的重要性分布
        alpha = F.softmax(scores_stack, dim=0) # [m, b, n, f]
        alpha_expanded = alpha.unsqueeze(3)  # [m, b, n, 1, f]
        
        # Stack weighted features
        f_weighted_stack = torch.stack(f_weighted_list, dim=0) #  [m, b, n, d, f]

        # 2. 生成“协同共识”信号 (Synergy Consensus Signal)：对所有模态加权求和 -> 得到当前系统中最强的频率特征组合
        f_consensus = torch.sum(alpha_expanded * f_weighted_stack, dim=0)  # [b, n, d, f]
        
        # =============================================================
        # Step 4: 反馈注入与重构 (Feedback & Reconstruction)
        # =============================================================
        f_consensus_expanded = f_consensus.unsqueeze(2) # [b, n, 1, d, f]
        
        # dynamic gamma:根据"原始信号"长什么样来决定是否需要协同
#         gamma = self.gamma_generator(f_amp.mean(dim=3)).unsqueeze(4)  # [b, n, m, 1, 1]
#         f_out_complex = f_complex + gamma * f_consensus_expanded  # [b, n, m, d, f]
        
        # 残差连接：原始信号 + 协同信号
        f_out_complex = f_complex + self.gamma * f_consensus_expanded  # [b, n, m, d, f]
        # no gamma
#         f_out_complex = f_complex + f_consensus_expanded  # [b, n, m, d, f]
        
        
        # IFFT 回到时域
        out_perm = torch.fft.irfft(f_out_complex, n=t, dim=-1)  # [b, n, m, d, t]
        out = out_perm.permute(0, 3, 1, 2, 4)  # [b, n, m, d, t] -> [b,d,n,m,t]    
        
        # save features
#         torch.save({
#             # ===== input =====
#             "h": h.detach().cpu(), # [b,d,n,m,t]
            
#             # ===== frequency domain =====
#             "f_complex": f_complex.detach().cpu(), # complex: [b,n,m,d,f]
#             "f_amp": f_amp.detach().cpu(), # amp: [b,n,m,d,f]
            
#             # ===== spatial heterogeneity =====
#             "node_emb": self.node_emb.detach().cpu(), # [n, d_node]
            
#             # ===== Modality-specific Frequency Filter =====
#             "weights": [w.detach().cpu() for w in w_list],  # m × [b,n,f]
#             "f_weighted_stack": f_weighted_stack.detach().cpu(), # weighted features: [m,b,n,d,f]
#             "scores_stack": scores_stack.detach().cpu(), # [m,b,n,f]           
            
#             # ===== Frequency-guided Synergy Augmenter =====
#             "alpha": alpha.detach().cpu(), # score after modality softmax: [m,b,n,f]
#             "gamma": self.gamma.detach().cpu().item(),
#             "f_consensus": f_consensus.detach().cpu(), # consensus
#             "f_out_complex": f_out_complex.detach().cpu(), # feature after feedback
#             "out_perm": out_perm.detach().cpu(), # IFFT feature
#         }, "/home/users/djw/MoFre/plot/features.pth")
        
        
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
        h_glu = self.temporal_glu(h)  # [b,2d,n,m,1]
        v, g = h_glu.chunk(2, dim=1)
        h = v * torch.sigmoid(g)

        pred = self.proj(h)  # [b,pred_len,n,m,1]
        return pred
    
    
class MoFre(nn.Module):
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
        super(MoFre, self).__init__()
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
        
def main():
    # 使用 argparse 更清晰地解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (default: 0)')
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        assert args.gpu < torch.cuda.device_count(), f"指定的 GPU:{args.gpu} 不存在，可用 GPU 数量为 {torch.cuda.device_count()}"
        device = torch.device(f'cuda:{args.gpu}')
        summary_device = 'cuda'  # torchsummary 只接受 'cuda' 或 'cpu'
    else:
        device = torch.device('cpu')
        summary_device = 'cpu'

    # 参数设置
    num_nodes = 98
    num_modes = 4
    d = 64
    batch = 8
    latents = 32
    seq_len = 16
    pred_len = 3
    layers = 4

    # 输入数据 [batch, seq_len, num_nodes]
    data = torch.rand(batch, seq_len, num_nodes, num_modes, 1).to(summary_device)
    print("data -> ", data.shape)
    # 模型实例化并移动到目标设备
    model = MoFre(seq_len, num_nodes, num_modes, pred_len, d, latents, layers).to(summary_device)

    # 打印模型结构
    summary(model, input_data=data, device=summary_device)

    # 测试前向传播
    out = model(data)
    print("out -> ", out.shape)

if __name__ == '__main__':
    main()        
                