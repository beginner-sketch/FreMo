import os
import h5py
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityWeightedLoss(nn.Module):
    def __init__(self, M, loss_type="mse", init_weights=None, learnable=False):
        """
        Args:
            M: 模态数
            loss_type: "mae" 或 "mse"
            init_weights: 初始权重 (list/ndarray/torch.Tensor)，长度=M
            learnable: 是否让权重可学习
        """
        super().__init__()
        self.M = M
        self.loss_type = loss_type
        
        if init_weights is None:
            init_weights = torch.tensor(torch.ones(M), dtype=torch.float32)

        if learnable:
            self.weights = nn.Parameter(init_weights)  # 可学习
        else:
            self.register_buffer("weights", init_weights)  # 固定

    def forward(self, pred, target):
        """
        Args:
            pred: [B, O, N, M, 1] 预测
            target: [B, O, N, M, 1] 标签
        Returns:
            加权 loss 标量
        """
        B, O, N, M, _ = pred.shape
        assert M == self.M, f"Expected M={self.M}, but got {M}"

        diff = pred - target
        if self.loss_type == "mae":
            loss_all = torch.abs(diff)
        elif self.loss_type == "mse":
            loss_all = diff**2
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # 每模态 loss: [M]
        # (B,O,N,M,_) → 按 B,O,N,_ 平均
        loss_per_modality = loss_all.mean(dim=(0,1,2,4))  # [M]

        # 加权
        weights = torch.relu(self.weights)  # 确保非负
        weights = weights / (weights.sum() + 1e-6)  # 归一化，避免无界增长
        weighted_loss = (loss_per_modality * weights).sum()

        return weighted_loss, loss_per_modality.detach(), weights.detach()


class UncertaintyLoss(nn.Module):
    def __init__(self, num_modalities):
        super(UncertaintyLoss, self).__init__()
        # 每个模态的 log variance 参数（初始化为 0，即 σ=1）
        self.log_vars = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, losses):
        """
        losses: list 或 tensor，包含每个模态的 loss 值
        e.g. losses = [loss_m1, loss_m2, loss_m3, loss_m4]
        """
        total_loss = 0
        for i, L_m in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # = 1/σ²
            total_loss += precision * L_m + 0.5 * self.log_vars[i]
        return total_loss
    
    
class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, n_source, day_slot, C_0=1):
    n_slot = day_slot

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route,  n_source, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            end = (i + offset) * day_slot + j + 1
            sta = end - n_frame
            if sta >= 0:
                tmp_seq[i * n_slot + j, :, :, :, :] = np.reshape(data_seq[sta:end, :, :], [n_frame, n_route, n_source, C_0])
    return tmp_seq

def data_gen(file_path, data_config, n_route, n_frame=21, n_source=4, day_slot=288):
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        h = h5py.File(file_path)
        data_seq = h["data"][:]
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    print("DATA SIZE: ", data_seq.shape)
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, n_source, day_slot)
    seq_train = seq_train[n_frame:]
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route,  n_source, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route,  n_source, day_slot)
    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.      
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    
    # scaler on each modality
#     mean = np.mean(seq_train, axis=(0,1,2,4)) 
#     std = np.std(seq_train, axis=(0,1,2,4))
#     x_stats = {'mean': mean, 'std': std}
#     x_train = z_score(seq_train, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
#     x_val = z_score(seq_val, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
#     x_test = z_score(seq_test, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
    
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False, period=None):
    len_inputs = len(inputs)
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)
    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        yield inputs[slide]
        
def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(x, mean, std):
    return x * std + mean

def RMSE(v, v_):
    return np.mean((v_ - v) ** 2, axis=(0, 2, 4)) ** 0.5

def MAE(v, v_):    
    return np.mean(np.abs(v_ - v), axis=(0, 2, 4))                                                                                                                                       

def get_metric(y, y_, x_stats):
#     y = z_inverse(y, x_stats['mean'], x_stats['std'])
#     y_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
    
    y = z_inverse(y, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
    y_ = z_inverse(y_, x_stats['mean'].reshape(1,1,1,-1,1), x_stats['std'].reshape(1,1,1,-1,1))
    
    rmse = RMSE(y, y_)
    mae = MAE(y, y_)
    return mae, rmse