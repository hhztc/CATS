import torch
import math
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding_inverted

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print(period_list, period_weight)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


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


# 定义GRU网络
class iGRU(nn.Module):
    def __init__(self, params):
        super(iGRU, self).__init__()
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.top_k = params['top_k']
        self.d_model = params['d_model']
        self.d_ff = params['d_ff']
        self.num_kernels = params['num_kernels']
        self.d_model = params['d_model']
        self.hidden_size = params['hidden_size']  # 隐层大小
        self.num_layers = params['num_layers']  # gru层数
        self.layer = params['e_layers']  # TimesBlocks层数
        self.feature_size = params['feature_size']
        self.e_layers = params['e_layers']
        self.num_class = params['num_class']
        self.embed = params['embed']
        self.freq = params['freq']
        self.dropout = params['dropout']
        self.enc_in = params['enc_in']

        ##  位置embedding
        self.person_embedding = nn.Linear(self.feature_size, self.d_model)
        self.position_embedding = PositionalEncoding(d_model=self.d_model, dropout=0.1)

        self.gru = nn.GRU(self.d_model, self.hidden_size, self.num_layers, batch_first=True)
        self.model = nn.ModuleList([TimesBlock(self.seq_len, self.pred_len, self.top_k, self.d_model, self.d_ff, self.num_kernels) for _ in range(self.e_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)
        # self.enc_embedding = DataEmbedding(params['enc_in'], params['d_model'], params['embed'], params['freq'], params['dropout'])
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq, self.dropout)
        self.act = F.gelu
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.num_class)
        self.sigmoid = nn.Sigmoid()
        print(1)

    def forward(self, x_enc, x_mark_enc):

        x = self.person_embedding(x_enc) + self.position_embedding(x_enc)

        enc_out, _ = self.gru(x)

        for i in range(self.layer):
            x = self.layer_norm(self.model[i](x))

        output = self.act(enc_out)
        output = output * x_mark_enc.unsqueeze(-1)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output

    # def forward(self, x_enc, x_mark_enc):
    #     x_enc = self.enc_embedding(x_enc, None)
    #     enc_out, _ = self.gru(x_enc)
    #     # for i in range(self.layer):
    #     #     enc_out = self.layer_norm(self.model[i](enc_out))
    #     output = self.act(enc_out)
    #     output = enc_out * x_mark_enc.unsqueeze(-1)
    #     output = self.fc(output[:, -1, :])
    #     output = self.sigmoid(output)
    #     return output


# iGru_params = {'seq_len': 7, 'pred_len': 0, 'top_k': 3, 'd_model': 16, 'd_ff': 32, 'num_kernels': 6, 'e_layers': 2,
#                'feature_size': 5, 'hidden_size': 32, 'num_layers': 2, 'num_class': 1, 'enc_in': 5, 'embed': "timeF",
#                'freq': 'd', 'dropout': 0.1}
#
# model = iGRU(iGru_params)
# input_tensor = torch.rand(53, 7, 5)
# padding_mask = torch.rand(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print("fdf", output_tensor)

