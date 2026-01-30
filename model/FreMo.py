class FreDoM(nn.Module):
    def __init__(self, seq_len, num_nodes, num_modes, d, node_emb_dim=32):
        super(FreDoM, self).__init__()
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
        
        # 4. 共识残差的缩放参数(对Consensus的听劝程度)
        self.gamma = nn.Parameter(torch.zeros(1)) 

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
        
        # 残差连接：原始信号 + 协同信号
        f_out_complex = f_complex + self.gamma * f_consensus_expanded  # [b, n, m, d, f]
        
        
        # IFFT 回到时域
        out_perm = torch.fft.irfft(f_out_complex, n=t, dim=-1)  # [b, n, m, d, t]
        out = out_perm.permute(0, 3, 1, 2, 4)  # [b, n, m, d, t] -> [b,d,n,m,t]    
        
        
        return out