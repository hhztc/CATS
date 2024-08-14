import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from math import ceil


class Crossformer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, params):
        super(Crossformer, self).__init__()
        self.seq_len = params['seq_len']
        self.pred_len = params['pred_len']
        self.e_layers = params['e_layers']
        self.output_attention = params['output_attention']
        self.enc_in = params['enc_in']
        self.d_model = params['d_model']
        self.embed = params['embed']
        self.freq = params['freq']
        self.num_class = params['num_class']
        self.factor = params['factor']
        self.n_heads = params['n_heads']
        self.d_ff = params['d_ff']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.win_size = params['win_size']
        self.seg_len = params['seg_len']
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (self.e_layers - 1)))
        self.head_nf = self.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(self.d_model, self.seg_len, self.seg_len, self.pad_in_len - self.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.in_seg_num, self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(self, 1 if l == 0 else self.win_size, self.d_model, self.n_heads, self.d_ff,
                            1, self.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), self.factor
                            ) for l in range(self.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, (self.pad_out_len // self.seg_len), self.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(self, (self.pad_out_len // self.seg_len), self.factor, self.d_model, self.n_heads,
                                           self.d_ff, self.dropout),
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    self.seg_len,
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    # activation=self.activation,
                )
                for l in range(self.e_layers + 1)
            ],
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.head_nf * self.enc_in, self.num_class)

    def forward(self, x_enc, x_mark_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        output = torch.sigmoid(output)
        return output
  