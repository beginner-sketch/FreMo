import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

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
        # Node embedding
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_emb_dim))

        input_dim = self.freq_dim + node_emb_dim
        self.weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d),
                nn.ReLU(),
                nn.Linear(d, self.freq_dim),
                nn.Sigmoid()
            ) for _ in range(num_modes)
        ])

        # Modality-shared Scalar
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, h):
        b, d, n, m, t = h.shape
        h_t = h.permute(0, 2, 3, 1, 4)
        # =============================================================
        # FFT over the time domain
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
            # Spectrum of modality i
            f_amp_m = f_amp[:, :, i, :, :]
            f_complex_m = f_complex[:, :, i, :, :]
            
            # Hidden dimension aggregation
            f_profile = f_amp_m.mean(dim=2)
            
            # Node embedding
            features = torch.cat([f_profile, node_emb_expanded], dim=-1)
            
            # Frequency weight generation
            w = self.weight_generators[i](features)
            
            # Frequency filter (Phase Preserving)
            w_expanded = w.unsqueeze(2)  
            f_weighted = f_complex_m * w_expanded
            
            # Score
            score = torch.abs(f_weighted).mean(dim=2)
            scores_list.append(score)

        # =============================================================
        # Frequency-Guided Synergy Integrator (FSI)
        # =============================================================
        # Frequency-wise synergy weight generation (modality softmax)
        scores_stack = torch.stack(scores_list, dim=0)
        alpha = F.softmax(scores_stack, dim=0)
        alpha_expanded = alpha.unsqueeze(3)
        
        # Filtered representation
        f_weighted_stack = torch.stack(f_weighted_list, dim=0)

        # Synergy consensus construction
        f_consensus = torch.sum(alpha_expanded * f_weighted_stack, dim=0)
        
        # Feedback and Reconstruction
        f_consensus_expanded = f_consensus.unsqueeze(2) 
        f_out_complex = f_complex + self.gamma * f_consensus_expanded

        # =============================================================
        # Back to the time domain
        # =============================================================        
        out_perm = torch.fft.irfft(f_out_complex, n=t, dim=-1)
        out = out_perm.permute(0, 3, 1, 2, 4)
        return out
