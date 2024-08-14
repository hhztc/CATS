import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=7):
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


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 计算查询、键、值
        query = self.query(x)
        key = self.key(x)
        value = x  # 值与输入相同

        # 计算注意力权重
        attn_weights = F.softmax(torch.matmul(query, key.transpose(1, 2)), dim=-1)

        # 加权求和得到上下文向量
        context = torch.matmul(attn_weights, value)

        return context.squeeze(2)


# 定义带注意力机制和位置编码的 GRU 模型
class AttentionGRU(nn.Module):
    def __init__(self, params):
        super(AttentionGRU, self).__init__()
        self.seq_len = params['seq_len']
        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']  # gru层数
        self.attention_dim = params['attention_dim']
        self.num_class = params['num_class']
        self.feature_size = params['feature_size']
        self.d_model = params['d_model']
        self.num_heads = params['num_heads']

        self.person_embedding = nn.Linear(self.feature_size, self.d_model, bias=False)
        self.position_embedding = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_len=self.seq_len)

        self.gru = nn.GRU(self.d_model, self.hidden_size, self.num_layers, batch_first=True)

        # 1------------------------
        # self.attention = SelfAttention(self.hidden_size)
        # 2------------------------
        # 定义注意力层
        # self.attention = nn.Linear(self.hidden_size, self.attention_dim)
        # self.context_vector = nn.Linear(self.attention_dim, self.num_class, bias=False)
        # 3------------------------
        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, batch_first=True, dropout=0.2)

        self.act = F.gelu
        self.fc = nn.Linear(self.hidden_size, out_features=1)  # 用于二分类的全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc):
        x = self.person_embedding(x_enc) + self.position_embedding(x_enc)
        # GRU 层
        output, hidden = self.gru(x)

        # 1-------------------------------
        # output = self.attention(output)
        # 2-------------------------------
        # 注意力机制
        # attention_weights = F.softmax(self.context_vector(torch.tanh(self.attention(output))), dim=1)
        # output = torch.bmm(attention_weights.transpose(1, 2), output)
        # 3-------------------------------
        # output, attn_output_weights = self.attention(output, output, output)

        output = self.act(output)

        output = output * x_mark_enc.unsqueeze(-1)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output


# iGru_params = {'seq_len': 14,  'd_model': 16, 'feature_size': 5, 'attention_dim': 10,
#                'hidden_size': 32, 'num_layers': 2, 'num_class': 1, 'num_heads': 8}
#
# model = AttentionGRU(iGru_params)
# input_tensor = torch.rand(53, 14, 5)
# padding_mask = torch.rand(53, 14)
# output_tensor = model(input_tensor, padding_mask)
# print("output_tensor", output_tensor)
