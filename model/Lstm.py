import torch.nn as nn
import torch
from layers.Embed import DataEmbedding


# 定义GRU网络
class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        self.hidden_size = params['hidden_size']  # 隐层大小
        self.num_layers = params['num_layers']  # gru层数
        self.feature_size = params['feature_size']
        self.num_class = params['num_class']
        self.enc_in = params['enc_in']
        self.d_model = params['d_model']
        self.embed = params['embed']
        self.freq = params['freq']
        self.dropout = params['dropout']

        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
        self.lstm = nn.LSTM(self.d_model, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_mark_enc):
        enc_out = self.enc_embedding(x, None)  # [B,T,C]
        output, _ = self.lstm(enc_out)
        output = output[:, -1, :]
        output = self.fc(output)
        output = self.sigmoid(output)
        return output

#
# lstm_params = {'feature_size': 5, 'hidden_size': 32, 'num_layers': 2, 'num_class': 1, 'enc_in': 5, 'd_model': 32,
#               'freq': 'd', 'embed': "timeF", 'dropout': 0.1}
# model = LSTM(lstm_params)
# input_tensor = torch.rand(53, 7, 5)
# padding_mask = torch.rand(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print(output_tensor.shape)