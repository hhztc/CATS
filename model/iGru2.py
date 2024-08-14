import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return self.dropout(x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(TimeFeatureEmbedding, self).__init__()
        self.embed = nn.Linear(d_feature, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#         self.register_buffer('weights', self.calculate_weights())
#
#     def calculate_weights(self):
#         weights = torch.zeros(self.max_len, self.d_model)
#         position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
#         time_decay = torch.exp(-position / 100)  # 可以调整时间衰减系数
#         weights[:, 0::2] = torch.sin(time_decay)
#         weights[:, 1::2] = torch.cos(time_decay)
#         return weights.unsqueeze(0)
#
#     def forward(self, x):
#         # 添加位置编码到输入张量中
#         x = x + self.weights[:, :x.size(1)]
#         return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层

        # 计算位置编码并将其存储在pe张量中
        pe = torch.zeros(max_len, d_model)  # 创建一个max_len x d_model的全零张量
        position = torch.arange(0, max_len).unsqueeze(1)  # 生成0到max_len-1的整数序列，并添加一个维度
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 引入时效性的权重，近位置权重更大
        time_decay = torch.arange(max_len, 0, -1, dtype=torch.float32).unsqueeze(1) / max_len
        pe *= time_decay
        pe = pe.unsqueeze(0)  # 在第一个维度添加一个维度，以便进行批处理
        self.register_buffer('pe', pe)  # 将位置编码张量注册为缓冲区，以便在不同设备之间传输模型时保持其状态

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x与对应的位置编码相加
        x = self.pe[:, :x.size(1)]
        # 应用dropout层并返回结果
        return self.dropout(x)


# 定义带注意力机制和位置编码的 GRU 模型
class iGRU(nn.Module):
    def __init__(self, params):
        super(iGRU, self).__init__()
        self.seq_len = params['seq_len']
        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']  # gru层数
        self.attention_dim = params['attention_dim']
        self.num_class = params['num_class']
        self.feature_size = params['feature_size']
        self.d_model = params['d_model']

        ##  位置embedding
        self.person_embedding = nn.Linear(self.feature_size, self.d_model)
        self.position_embedding = PositionalEncoding(d_model=self.d_model, dropout=0.1)
        self.gru = nn.GRU(self.d_model, self.hidden_size, self.num_layers, batch_first=True)

        self.act = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.hidden_size, out_features=self.num_class)  # 用于二分类的全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc):
        x = self.person_embedding(x_enc) + self.position_embedding(x_enc)
        # TimesBlock
        for i in range(self.layer):
            x = self.layer_norm(self.model[i](x))
        # GRU 层
        output, _ = self.gru(x)

        output = output * x_mark_enc.unsqueeze(-1)
        # output = self.act(context)
        output = self.dropout(output)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output


#
iGru_params = {'seq_len': 7,  'd_model': 32, 'feature_size': 5, 'attention_dim': 10,
               'hidden_size': 32, 'num_layers': 2, 'num_class': 1}

model = iGRU(iGru_params)
input_tensor = torch.rand(53, 7, 5)
padding_mask = torch.rand(53, 7)
output_tensor = model(input_tensor, padding_mask)
print("output_tensor", output_tensor)
