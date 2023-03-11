

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy


def attention(Q, K, V, mask=None, dropout=None):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)
    d_k = Q.size(-1)
    # (B, H, S, S)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)

    if dropout is not None:
        out = dropout(out)
    # (B, *(H), seq_len, d_model//H = d_k)
    return out


class MultiheadedAttention(nn.Module):
    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None, d_out=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_out = d_out
        self.d_out = d_model_Q if d_out is None else d_out
        self.d_model = d_model_Q if d_model is None else d_model
        self.d_k = self.d_model // H
        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_out)
        self.dropout = nn.Dropout(self.dout_p)
        self.visual = False
        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask=None):
        B, Sq, _ = Q.shape
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)
        # (B, H, Sm, d_k) <- (B, Sm, D)
        # (-4, -3*, -2*, -1)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)
        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        Q = attention(Q, K, V, mask, self.dropout)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        Q = self.linear_d2Q(Q)
        return Q


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def generate_sincos_embedding(seq_len, d_model, train_len=None):
    odds = np.arange(0, d_model, 2)
    evens = np.arange(1, d_model, 2)
    pos_enc_mat = np.zeros((seq_len, d_model))
    pos_list = np.arange(seq_len) if train_len is None else np.linspace(0, train_len-1, num=seq_len)
    for i, pos in enumerate(pos_list):
        pos_enc_mat[i, odds] = np.sin(pos / (10000 ** (odds / d_model)))
        pos_enc_mat[i, evens] = np.cos(pos / (10000 ** (evens / d_model)))
    return torch.from_numpy(pos_enc_mat).unsqueeze(0)


class PositionalEncoder(nn.Module):
    def __init__(self, cfg, dout_p):
        super(PositionalEncoder, self).__init__()
        self.seq_len = cfg.TRAIN.NUM_FRAMES
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x):
        B, S, d_model = x.shape
        if S != self.seq_len:
            pos_enc_mat = generate_sincos_embedding(S, d_model, self.seq_len)
            x = x + pos_enc_mat.type_as(x)
        else:
            pos_enc_mat = generate_sincos_embedding(S, d_model)
            x = x + pos_enc_mat.type_as(x)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer):
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        return x + res


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dout_p):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dout_p, H=8, d_ff=None, d_hidden=None):
        super(EncoderLayer, self).__init__()
        self.res_layer0 = ResidualConnection(d_model, dout_p)
        self.res_layer1 = ResidualConnection(d_model, dout_p)
        d_hidden = d_model if d_hidden is None else d_hidden
        d_ff = 4*d_model if d_ff is None else d_ff
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H, d_model=d_hidden)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask=None):
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        def sublayer0(x): 
            return self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        x = self.res_layer0(x, sublayer0)
        x = self.res_layer1(x, sublayer1)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, dout_p, H, d_ff, N, d_hidden=None):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff, d_hidden), N)

    def forward(self, x, src_mask=None):
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x


class TransformerEmbModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.MODEL.BASE_MODEL.OUT_CHANNEL
        fc_params = cfg.MODEL.EMBEDDER_MODEL.FC_LAYERS
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        cap_scalar = cfg.MODEL.EMBEDDER_MODEL.CAPACITY_SCALAR
        hidden_channels = cfg.MODEL.EMBEDDER_MODEL.HIDDEN_SIZE
        self.embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE

        self.pooling = nn.AdaptiveMaxPool2d(1)
        fc_layers = []
        for channels, _ in fc_params:
            channels = channels * cap_scalar
            fc_layers.append(nn.Dropout(drop_rate))
            fc_layers.append(nn.Linear(in_channels, channels))
            fc_layers.append(nn.BatchNorm1d(channels))
            fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*fc_layers)
        self.video_emb = nn.Linear(in_channels, hidden_channels)
        self.video_pos_enc = PositionalEncoder(cfg, drop_rate)
        self.video_encoder = Encoder(hidden_channels, drop_rate, cfg.MODEL.EMBEDDER_MODEL.NUM_HEADS, 
                                     cfg.MODEL.EMBEDDER_MODEL.D_FF, cfg.MODEL.EMBEDDER_MODEL.NUM_LAYERS)
        self.embedding_layer = nn.Linear(hidden_channels, self.embedding_size)

    def forward(self, x, video_masks=None):
        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.video_emb(x)
        x = x.view(batch_size, num_steps, x.size(1))
        x = self.video_pos_enc(x)
        x = self.video_encoder(x, src_mask=video_masks)
        x = x.view(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x


class MLPHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        projection_hidden_size = cfg.MODEL.PROJECTION_SIZE
        embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE
        self.net = nn.Sequential(nn.Linear(embedding_size, projection_hidden_size),
                                 nn.BatchNorm1d(projection_hidden_size),
                                 nn.ReLU(True),
                                 nn.Linear(projection_hidden_size, embedding_size))

    def forward(self, x):
        b, l, c = x.shape
        x = x.view(-1, c)
        x = self.net(x)
        return x.view(b, l, c)


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        num_classes = cfg.EVAL.CLASS_NUM
        drop_rate = cfg.MODEL.EMBEDDER_MODEL.FC_DROPOUT_RATE
        embedding_size = cfg.MODEL.EMBEDDER_MODEL.EMBEDDING_SIZE

        fc_layers = []
        fc_layers.append(nn.Dropout(drop_rate))
        fc_layers.append(nn.Linear(embedding_size, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc_layers(x)


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        res50_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.backbone = nn.Sequential(*list(res50_model.children())[:-3])
        self.res_finetune = list(res50_model.children())[-3]
        self.embed = TransformerEmbModel(cfg)
        self.ssl_projection = MLPHead(cfg)
        self.classifier = Classifier(cfg)

    def forward(self, x, video_masks=None, project=False):
        batch_size, num_steps, c, h, w = x.shape
        frames_per_batch = self.cfg.MODEL.BASE_MODEL.FRAMES_PER_BATCH
        num_blocks = int(math.ceil(float(num_steps)/frames_per_batch))
        backbone_out = []
        for i in range(num_blocks):
            curr_idx = i * frames_per_batch
            cur_steps = min(num_steps-curr_idx, frames_per_batch)
            curr_data = x[:, curr_idx:curr_idx+cur_steps]
            curr_data = curr_data.contiguous().view(-1, c, h, w)
            self.backbone.eval()
            with torch.no_grad():
                curr_emb = self.backbone(curr_data)
            curr_emb = self.res_finetune(curr_emb)
            _, out_c, out_h, out_w = curr_emb.size()
            curr_emb = curr_emb.contiguous().view(batch_size, cur_steps, out_c, out_h, out_w)
            backbone_out.append(curr_emb)
        x = torch.cat(backbone_out, dim=1)
        x = self.embed(x, video_masks=video_masks)
        if self.cfg.MODEL.PROJECTION and project:
            x = self.ssl_projection(x)
        x = F.normalize(x, dim=-1)
        return self.classifier(x)


if __name__ == '__main__':
    from config import get_cfg
    cfg = get_cfg()
    model = TransformerModel(cfg)
    print(model)
