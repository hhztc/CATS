# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, dropout=0.1):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(
#                     m.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return self.dropout(x)
#
#
# # 定义带注意力机制和位置编码的 GRU 模型
# class iGRU(nn.Module):
#     def __init__(self, params):
#         super(iGRU, self).__init__()
#         self.seq_len = params['seq_len']
#         self.hidden_size = params['hidden_size']
#         self.num_layers = params['num_layers']  # gru层数
#         self.attention_dim = params['attention_dim']
#         self.num_class = params['num_class']
#         self.feature_size = params['feature_size']
#         self.d_model = params['d_model']
#
#         self.enc_embedding = TokenEmbedding(self.feature_size, self.d_model)
#         self.gru = nn.GRU(self.d_model, self.hidden_size, self.num_layers, batch_first=True)
#         self.attention = nn.Linear(self.hidden_size, self.attention_dim)
#         self.context_vector = nn.Linear(self.attention_dim, self.num_class, bias=False)
#
#         self.act = F.gelu
#         self.fc = nn.Linear(self.hidden_size, out_features=1)  # 用于二分类的全连接层
#         self.sigmoid = nn.Sigmoid()
#
#     def positional_encoding(self, x):
#         seq_len = x.size(1)
#         encoding = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0) / seq_len
#         encoding = encoding.unsqueeze(-1)
#         encoding = encoding.expand(x.size())
#         return encoding
#
#     def forward(self, x_enc, x_mark_enc):
#         x = self.enc_embedding(x_enc)  # [B,T,C]
#         # 添加位置编码
#         x = x + self.positional_encoding(x)
#         # GRU 层
#         output, hidden = self.gru(x)
#
#         # 注意力机制
#         attention_weights = F.softmax(self.context_vector(torch.tanh(self.attention(output))), dim=1)
#
#         context = torch.bmm(attention_weights.transpose(1, 2), output)
#         output = self.act(context)
#         output = output * x_mark_enc.unsqueeze(-1)
#         output = self.fc(output[:, -1, :])
#         output = self.sigmoid(output)
#
#         return output
#
#
# iGru_params = {'seq_len': 7,  'd_model': 16, 'feature_size': 5, 'attention_dim': 10,
#                'hidden_size': 32, 'num_layers': 2, 'num_class': 1}
#
# model = iGRU(iGru_params)
# input_tensor = torch.rand(53, 7, 5)
# padding_mask = torch.rand(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print("output_tensor", output_tensor)
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class SelfAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(SelfAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         # 计算查询、键、值
#         query = self.query(x)
#         key = self.key(x)
#         value = x  # 值与输入相同
#
#         # 计算注意力权重
#         attn_weights = F.softmax(torch.matmul(query, key.transpose(1, 2)), dim=-1)
#
#         # 加权求和得到上下文向量
#         context = torch.matmul(attn_weights, value)
#
#         return context.squeeze(2)
#
#
# class AttentionGRUWithBinaryClassification(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(AttentionGRUWithBinaryClassification, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.attention = SelfAttention(hidden_size)
#         self.fc = nn.Linear(hidden_size, 1)  # 用于二分类任务的全连接层
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         output, _ = self.gru(x)
#
#         # 注意力机制
#         context = self.attention(output)
#
#         # 二分类任务
#         output = self.fc(context)
#         output = self.sigmoid(output)
#
#         return output
#
#
# # 示例使用
# input_size = 5
# hidden_size = 20
# num_layers = 2
# seq_length = 7
# batch_size = 32
#
# model = AttentionGRUWithBinaryClassification(input_size, hidden_size, num_layers)
# input_data = torch.randn(batch_size, seq_length, input_size)
# output = model(input_data)
#
# print("Output shape:", output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionWithPositionalEncoding(nn.Module):
    def __init__(self, input_size, attention_size):
        super(SelfAttentionWithPositionalEncoding, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size

        # 定义查询、键和值的线性变换层
        self.linear_q = nn.Linear(input_size, attention_size)
        self.linear_k = nn.Linear(input_size, attention_size)
        self.linear_v = nn.Linear(input_size, attention_size)

        # 生成位置编码
        self.positional_encoding = self.generate_positional_encoding(attention_size)

    def generate_positional_encoding(self, attention_size):
        pe = torch.zeros(attention_size)
        position = torch.arange(0, attention_size, dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, attention_size, 2, dtype=torch.float32) * -(
                    torch.log(torch.tensor(10000.0)) / attention_size))
        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        seq_length = x.size(1)

        # 使用线性变换将输入映射到查询、键和值空间
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # 添加位置编码
        q = q + self.positional_encoding.unsqueeze(0)
        k = k + self.positional_encoding.unsqueeze(0)

        # 计算注意力权重
        attention_weights = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_size, dtype=torch.float32)),
            dim=-1)

        # 将注意力权重乘以值，并对所有位置进行加权求和得到输出
        output = torch.matmul(attention_weights, v)

        return output


# 示例使用
input_size = 10
attention_size = 5
seq_length = 7
batch_size = 32

attention = SelfAttentionWithPositionalEncoding(input_size, attention_size)
input_data = torch.randn(batch_size, seq_length, input_size)
output = attention(input_data)

print("Output shape:", output.shape)


