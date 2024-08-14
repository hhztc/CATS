import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, params):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(PatchTST, self).__init__()
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.label_len = params['label_len']
        self.enc_in = params['enc_in']
        self.d_model = params['d_model']
        self.pred_len = params['pred_len']
        self.embed = params['embed']
        self.freq = params['freq']
        self.dropout = params['dropout']
        self.factor = params['factor']
        self.dropout = params['dropout']
        self.output_attention = params['output_attention']
        self.d_ff = params['d_ff']
        self.activation = params['activation']
        self.e_layers = params['e_layers']
        self.activation = params['activation']
        self.n_heads = params['n_heads']
        self.d_layers = params['d_layers']
        self.num_class = params['num_class']
        self.dec_in = params['dec_in']
        self.distil = params['distil']
        self.patch_len = params['patch_len']
        self.stride = params['stride']
        self.padding = self.stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.padding, self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Prediction Head
        self.head_nf = self.d_model * \
        int((self.seq_len - self.patch_len) / self.stride + 2)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(
            self.head_nf * self.enc_in, self.num_class)

    def forward(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        output = torch.sigmoid(output)
        return output


# PatchTST_params = {'seq_len': 7, 'pred_len': 0, 'label_len': 48, 'e_layers': 3, 'output_attention': False, 'enc_in': 5, 'd_model': 128,
#                    'embed': "timeF", 'freq': 'd', 'num_class': 1, 'factor': 1, 'n_heads': 8, 'd_ff': 256,
#                    'activation': 'gelu', 'dropout': 0.1, 'distil': False, 'd_layers': 1, 'dec_in': 5, 'patch_len': 15, 'stride': 8}
#
# model = PatchTST(PatchTST_params)
#
# input_tensor = torch.Tensor(53, 7, 5)
# padding_mask = torch.Tensor(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print(output_tensor)
