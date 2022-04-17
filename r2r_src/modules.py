import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_mlp(input_dim, hidden_dims, output_dim=None,
              use_batchnorm=False, dropout=0, fc_bias=True, relu=True):
    layers = []
    D = input_dim
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim, bias=fc_bias))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            if relu:
                layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim, bias=fc_bias))
    return nn.Sequential(*layers)


class SoftAttention(nn.Module):
    """Soft-Attention without learnable parameters
    """

    def __init__(self):
        super(SoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, proj_context, context=None, mask=None, reverse_attn=False):
        """Propagate h through the network.
        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        # Get attention
        attn = torch.bmm(proj_context, h.unsqueeze(2)).squeeze(2)  # batch x seq_len

        if reverse_attn:
            attn = -attn

        if mask is not None:
            attn.data.masked_fill_((mask == 0).data, -float('inf'))
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        if context is not None:
            weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        else:
            weighted_context = torch.bmm(attn3, proj_context).squeeze(1)  # batch x dim

        return weighted_context, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None, reverse_attn=False):
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if reverse_attn:
            attn = -attn

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn)
            attn.data.masked_fill_((attn_mask == 0).data, -float('inf'))
            # attn = attn.masked_fill((attn_mask == 0).data, -np.inf)

        attn_weight = self.softmax(attn.view(-1, len_k)).view(-1, len_q, len_k)

        attn_weight = self.dropout(attn_weight)
        output = torch.bmm(attn_weight, v)
        return output, attn_weight


class PositionalEncoding(nn.Module):
    """Implement the PE function to introduce the concept of relative position"""

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    """Implement the PE function to introduce the concept of relative position"""

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)



class StateAttention(nn.Module):
    def __init__(self):
        super(StateAttention, self).__init__()
        self.sm = nn.Softmax(dim=1)

    def forward(self, a_t, r_t, input_embedding, padded_mask, step):
        new_a_t = torch.zeros_like(a_t)
        for i in range(a_t.shape[1]):
            if i==0:
                new_a_t[:,i] = a_t[:,0]*r_t[:,0]
            else:
                window = a_t[:,i-1:i+1]
                window_sum = window[:,0]*r_t[:,1] + window[:,1]*r_t[:,0]

                new_a_t[:,i-1] += (1-padded_mask[:,i]) * (window_sum)
                new_a_t[:,i] += (padded_mask[:,i]) * (window_sum)

        new_a_t = new_a_t.unsqueeze(dim=1)
        output = torch.matmul(new_a_t, input_embedding).squeeze(dim=1)
        return output, new_a_t.squeeze(dim=1)


class ConfigObjAttention(nn.Module):
    def __init__(self):
        super(ConfigObjAttention, self).__init__()
        self.sm = nn.Softmax(dim=2)
    
    def forward(self, config_feature, image_feature, atten_mask, object_mask):
        # atten: 4 x 1 x 128 
        # image_weight batch x 576 x 128
        # atten_mask batch x 16
        # logit: 4 x 16 x 36
        batch_size, navi_nums, object_num = object_mask.shape
        atten_weight = config_feature.unsqueeze(dim=1) # 4 x 1 x128
        atten_weight = torch.bmm(atten_weight, torch.transpose(image_feature, 1, 2)).squeeze(dim=1)# 4 x 576
        atten_weight = atten_weight.view(batch_size, navi_nums, object_num) # 4 x 16 x 36
        atten_mask = atten_mask.unsqueeze(dim=2)
        tmp_atten_object_mask = atten_mask.repeat(1,3,1) * object_mask
        extened_padded_mask = ((1.0 - tmp_atten_object_mask) * -1e9)
        atten_weight = atten_weight + extened_padded_mask
        atten_weight = self.sm(atten_weight) # 4 x 16 x 36
        atten_weight = atten_weight.unsqueeze(dim=2)
        image_feature = image_feature.view(batch_size, navi_nums, object_num, 128)
        weighted_config_img_feat = torch.matmul(atten_weight, image_feature).squeeze(dim=2) # 4 x 16 x 1 x 128

        return weighted_config_img_feat, atten_weight.squeeze(dim=2)

def create_mask(batchsize, max_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]] = 1
    return tensor_mask.to(device)

def create_mask_for_object(batchsize, max_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]*36] = 1
    return tensor_mask.to(device)

def proj_masking(feat, projector, mask=None):
    """Universal projector and masking"""
    proj_feat = projector(feat.view(-1, feat.size(2)))
    proj_feat = proj_feat.view(feat.size(0), feat.size(1), -1)
    if mask is not None:
        return proj_feat * mask.unsqueeze(2).expand_as(proj_feat)
    else:
        return proj_feat