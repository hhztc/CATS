import math
import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


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

# 定义GRU网络
class GRU(nn.Module):
    def __init__(self, params):
        super(GRU, self).__init__()
        self.hidden_size = params['hidden_size']  # 隐层大小
        self.num_layers = params['num_layers']  # gru层数
        self.feature_size = params['feature_size']  # 特征数
        self.num_class = params['num_class']
        self.enc_in = params['enc_in']
        self.seq_len = params['seq_len']
        self.d_model = params['d_model']
        self.embed = params['embed']
        self.freq = params['freq']
        self.dropout = params['dropout']

        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
        self.person_embedding = nn.Linear(self.feature_size, self.d_model, bias=False)
        self.position_embedding = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_len=self.seq_len)
        self.gru = nn.GRU(self.d_model, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_mark_enc):
        # enc_out = self.enc_embedding(x, None)  # [B,T,C]
        # x = self.person_embedding(x) + self.position_embedding(x)
        x = self.person_embedding(x)
        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output


# gru_params = {'seq_len': 7, 'feature_size': 5, 'hidden_size': 32, 'num_layers': 2, 'num_class': 1, 'enc_in': 5, 'd_model': 32,
#               'freq': 'd', 'embed': "timeF", 'dropout': 0.1}
# model = GRU(gru_params)
# input_tensor = torch.rand(53, 7, 5)
# padding_mask = torch.rand(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print(output_tensor.shape)

    # def forward(self, x, hidden=None):
    #     batch_size = x.shape[0]  # 获取批次大小
    #
    #     # 初始化隐层状态
    #     if hidden is None:
    #         h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
    #     else:
    #         h_0 = hidden
    #
    #     # GRU运算
    #     output, h_0 = self.gru(x, h_0)
    #
    #     # 获取GRU输出的维度信息
    #     batch_size, timestep, hidden_size = output.shape
    #
    #     # 将output变成 batch_size * timestep, hidden_dim
    #     output = output.reshape(-1, hidden_size)
    #
    #     # 全连接层
    #     output = self.fc(output)  # 形状为batch_size * timestep, 1
    #
    #     output = self.sigmoid(output)
    #
    #     # 转换维度，用于输出
    #     output = output.reshape(timestep, batch_size, -1)
    #
    #     # 我们只需要返回最后一个时间片的数据即可
    #     return output[-1]
