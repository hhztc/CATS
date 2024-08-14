import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, params):
        super(iTransformer, self).__init__()
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
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq, self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model, self.d_ff, dropout=self.dropout, activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.enc_in, self.num_class)

    def forward(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        output = torch.sigmoid(output)
        return output

# #
# iTransformer_params = {'seq_len': 7, 'pred_len': 0, 'e_layers': 3, 'output_attention': False, 'enc_in': 5, 'd_model': 32,
#                        'embed': "timeF", 'freq': 'd', 'num_class': 1, 'factor': 1, 'n_heads': 8, 'd_ff': 64,
#                        'activation': 'gelu', 'dropout': 0.1}
# #
# model = iTransformer(iTransformer_params)
# input_tensor = torch.Tensor(53, 7, 5)
# padding_mask = torch.Tensor(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print(output_tensor.shape)
