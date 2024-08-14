# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class LSTNet(nn.Module):
#     def __init__(self, feature_size: int):
#         super(LSTNet, self).__init__()
#         self.P = 7
#         self.m = feature_size
#         self.hidR = 64  # Rnn隐藏层
#         self.hidC = 53  # args.hidCNN;
#         self.hidS = 5  # args.hidSkip
#         self.Ck = 7  # args.CNN_kernel
#         self.skip = 1  # args.skip
#         self.pt = 1  # int((self.P - self.Ck) / self.skip)
#         self.hw = 7  # args.highway_window
#
#         self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
#         self.GRU1 = nn.GRU(self.hidC, self.hidR)
#         self.dropout = nn.Dropout(p=0.2, inplace=False)
#         self.GRU_skip = nn.GRU(self.hidC, self.hidS)
#         self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
#         self.highway = nn.Linear(self.hw, 1)
#         self.output = nn.Sigmoid()
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         # CNN
#         c = x.view(-1, 1, self.P, self.m)  # shape(530, 7, 5) -> shape(530, 1 , 7, 5)
#         c = self.conv1(c)
#         c = F.relu(c)  # shape(530, 100, 2, 1)
#         c = self.dropout(c)
#         c = torch.squeeze(c, 3)   # shape(530, 100 ,2, 1) -> shape(530, 100 , 2) 去掉最后一个维度
#
#         # RNN
#         r = c.permute(2, 0, 1).contiguous()  # shape(530, 100 , 2 ) -> shape (2, 530 , 100)
#         _, r = self.GRU1(r)
#         r = self.dropout(torch.squeeze(r, 0))
#
#         # skip-rnn
#
#         if self.skip > 0:
#             s = c[:, :, int(-self.pt * self.skip):].contiguous()
#             print(s.shape)
#             s = s.reshape(batch_size, self.hidC, self.pt, self.skip)
#             s = s.permute(2, 0, 3, 1).contiguous()
#             s = s.view(self.pt, batch_size * self.skip, self.hidC)
#             _, s = self.GRU_skip(s)
#             s = s.view(batch_size, self.skip * self.hidS)
#             s = self.dropout(s)
#             r = torch.cat((r, s), 1)
#
#         res = self.linear1(r)
#         print(res.shape)
#
#         # # highway
#         if self.hw > 0:
#             z = x[:, -self.hw:, :]
#             print(z.shape)
#             z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
#             z = self.highway(z)
#             # z = z.view(-1, self.m)
#             z = z.view(-1, 1)
#             res = res + z
#
#         res = self.output(res)
#
#         print(res.shape)
#
#         return res[-1]
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNet(nn.Module):
    def __init__(self, params):
        super(LSTNet, self).__init__()
        self.enc_in = params['enc_in']
        self.cnn_channel = params['cnn_channel']
        self.hidden_size = params['hidden_size']
        self.num_kernels = params['num_kernels']
        self.dropout = params['dropout']
        self.num_class = params['num_class']
        self.conv = nn.Conv2d(1, self.cnn_channel, kernel_size=(self.num_kernels, self.enc_in))
        self.gru = nn.GRU(self.cnn_channel, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.num_class)  # Change output_dim to 1
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x, x_mark_enc):
        # Convolution
        c = self.conv(x.unsqueeze(1))  # Add channel dimension
        c = F.relu(c.squeeze(3))  # Remove last dimension

        # GRU
        r, _ = self.gru(c.permute(0, 2, 1))  # Swap time and feature dimensions
        r = self.dropout(r)

        # Linear
        y = self.linear(r[:, -1, :])

        # Sigmoid
        y = torch.sigmoid(y)

        return y

#
# model = LSTNet(input_dim=5, hidden_dim=64, kernel_size=6, dropout_rate=0.2)
# x = torch.Tensor(530, 7, 5)
# y = model(x)
# print(y.shape)

