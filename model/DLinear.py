import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecompose(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecompose, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, params):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(DLinear, self).__init__()
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.enc_in = params['enc_in']
        self.kernel_size = params['kernel_size']
        self.num_class = params['num_class']
        # Series decomposition block from Autoformer
        self.decompose = SeriesDecompose(kernel_size=self.kernel_size)
        self.channels = self.enc_in

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.act = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.projection = nn.Linear(self.enc_in * self.seq_len, self.num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_mark_enc):
        seasonal_init, trend_init = self.decompose(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output

        output = x.reshape(x.shape[0], -1)

        output = self.projection(output)

        output = self.sigmoid(output)

        return output


# model = DLinear(seq_len=7, pred_len=7, enc_in=5, individual=False)
# input_tensor = torch.Tensor(53, 7, 5)
# padding_mask = torch.Tensor(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print(output_tensor)
