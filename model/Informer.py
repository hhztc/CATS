import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, params):
        super(Informer, self).__init__()
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

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers - 1)
            ] if self.distil and ('forecast' in self.task_name) else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.num_class, bias=True)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.seq_len, self.num_class)

    def forward(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        output = torch.sigmoid(output)
        return output


# Informer_params = {'seq_len': 7, 'pred_len': 0, 'label_len': 48, 'e_layers': 3, 'output_attention': False, 'enc_in': 5, 'd_model': 64,
#                    'embed': "timeF", 'freq': 'd', 'num_class': 1, 'factor': 1, 'n_heads': 8, 'd_ff': 256,
#                    'activation': 'gelu', 'dropout': 0.1, 'distil': False, 'd_layers': 1, 'dec_in': 5}
# #
# model = Informer(Informer_params)
# input_tensor = torch.Tensor(53, 7, 5)
# padding_mask = torch.Tensor(53, 7)
# output_tensor = model(input_tensor, padding_mask)
# print(output_tensor)
