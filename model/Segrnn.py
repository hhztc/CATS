import torch
import torch.nn as nn
from layers.RevIN import RevIN


class SegRnn(nn.Module):
    def __init__(self, params):
        super(SegRnn, self).__init__()
        # get parameters
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.enc_in = params['enc_in']
        self.d_model = params['d_model']
        self.dropout = params['dropout']
        self.seg_len = params['seg_len']
        self.channel_id = params['channel_id']
        self.revin = params['revin']
        self.num_layers = params['num_layers']

        self.seg_num_x = self.seq_len//self.seg_len

        # build model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.num_layers, bias=True,
                          batch_first=True, bidirectional=False)

        self.seg_num_y = self.pred_len // self.seg_len

        if self.channel_id:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
            self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
        else:
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)

        self.fc = nn.Linear(self.enc_in, out_features=1)  # 用于二分类的全连接层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_mask):

        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1) # b,c,s

        x = x.reshape(-1, self.seg_num_x, self.seg_len)

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding

        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # decoding

        if self.channel_id:
            # m,d//2 -> 1,m,d//2 -> c,m,d//2
            # c,d//2 -> c,1,d//2 -> c,m,d//2
            # c,m,d -> cm,1,d -> bcm, 1, d
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
        else:
            # m,d -> bcm,d -> bcm, 1, d
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

        # pos_emb: m,d -> bcm,d ->  bcm,1,d
        # hn, cn: 1,bc,d -> 1,bc,md -> 1,bcm,d
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        output = self.fc(y[:, -1, :])

        output = self.sigmoid(output)

        return output


SegRnn_params = {'seq_len': 7,  'd_model': 32,  'pred_len': 10, 'enc_in': 5, 'dropout': 0.1,
                 'num_layers': 1, 'seg_len': 1, 'num_class': 1, 'revin': 1, 'channel_id': 0}

# model = SegRnn(SegRnn_params)
# input_tensor = torch.rand(53, 7, 5)
# padding_mask = torch.rand(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print("output_tensor", output_tensor)
