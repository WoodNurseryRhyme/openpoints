'''
Author: Mu Tongyao
Date: 2022-12-19 15:09:54
LastEditTime: 2023-03-15 11:23:15
LastEditors: Mu Tongyao
Description: 
FilePath: /topology_aware_completion/openpoints/models/backbone/tac_encoder.py
'''

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from pointnet2_ops.pointnet2_utils import (ball_query, furthest_point_sample,
                                           gather_operation,
                                           grouping_operation,
                                           three_interpolate)
from torch import einsum

from ..build import MODELS



def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    _, idx = sqrdists.topk(nsample, largest=False)
    return idx.int()

def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)

    #TODO：尝试将KNn的选择方式从距离改为余弦相似性，
    _, group_idx = dist.topk(k, largest=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def group_local(xyz, k=20, return_idx=False):
    """
    Input:
        x: point cloud, [B, 3, N]
    Return:
        group_xyz: [B, 3, N, K]
    """
    xyz = xyz.transpose(2, 1).contiguous()
    idx = query_knn_point(k, xyz, xyz)
    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz.permute(0, 3, 1, 2)
    if return_idx:
        return group_xyz, idx

    return group_xyz

class EdgeConv(torch.nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C2, N]
    """

    def __init__(self, input_channel, output_channel, k):
        super(EdgeConv, self).__init__()
        self.num_neigh = k

        self.conv = nn.Sequential(
            nn.Conv2d(2 * input_channel, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(output_channel // 2, output_channel // 2, kernel_size=1),
            nn.BatchNorm2d(output_channel // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(output_channel // 2, output_channel, kernel_size=1)
        )

    def forward(self, inputs):
        batch_size, dims, num_points = inputs.shape
        if self.num_neigh is not None:
            neigh_feature = group_local(inputs, k=self.num_neigh).contiguous()
            central_feat = inputs.unsqueeze(dim=3).repeat(1, 1, 1, self.num_neigh)
        else:
            central_feat = torch.zeros(batch_size, dims, num_points, 1).to(inputs.device)
            neigh_feature = inputs.unsqueeze(-1)
        edge_feature = central_feat - neigh_feature
        feature = torch.cat((edge_feature, central_feat), dim=1)
        feature = self.conv(feature)
        central_feature = feature.max(dim=-1, keepdim=False)[0]
        return central_feature


class AdaptGraphPooling(nn.Module):
    def __init__(self, pooling_rate, in_channel, neighbor_num, dim=64):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(dim, 3 + in_channel, 1)
        )

    def forward(self, vertices, feature_map, idx=False):
        """
        Input:
            vertices: "(bs, 3, vertice_num)",
            feature_map: "(bs, channel_num, vertice_num)",
        Return:
            vertices_pool: (bs, 3, pool_vertice_num),
            feature_map_pool: (bs, channel_num, pool_vertice_num)
        """

        bs, _, vertice_num = vertices.size()
        new_npoints = int(vertice_num*1.0 / self.pooling_rate+0.5)
        key_points_idx = furthest_point_sample(vertices.transpose(2,1).contiguous(), new_npoints)
        key_point = gather_operation(vertices, key_points_idx)
        key_feat = gather_operation(feature_map, key_points_idx)

        key_point_idx = query_knn(self.neighbor_num, vertices.transpose(2,1).contiguous(), key_point.transpose(2,1).contiguous(), include_self=True)

        group_point = grouping_operation(vertices, key_point_idx)
        group_feat = grouping_operation(feature_map, key_point_idx)

        qk_rel = key_feat.reshape((bs, -1, new_npoints, 1)) - group_feat
        pos_rel = key_point.reshape((bs, -1, new_npoints, 1)) - group_point

        pos_embedding = self.pos_mlp(pos_rel)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding) # b, in_channel + 3, n, n_knn
        sample_weight = torch.softmax(sample_weight, -1) # b, in_channel + 3, n, n_knn
        new_xyz_weight = sample_weight[:,:3,:,:]  # b, 3, n, n_knn
        new_feture_weight = sample_weight[:,3:,:,:]  # b, in_channel, n, n_knn

        group_feat = group_feat + pos_embedding  #
        new_feat = einsum('b c i j, b c i j -> b c i', new_feture_weight, group_feat)
        new_point = einsum('b c i j, b c i j -> b c i', new_xyz_weight, group_point)

        return new_point, new_feat


@MODELS.register_module()
class TACEncoder(nn.Module):

    def __init__(self, 
                 in_channels: int,
                 emb_dims: int,
                 **kwargs):
        super(TACEncoder, self).__init__()

        self.out_channel = emb_dims//2

        # HGNet econder
        self.gcn_1 = EdgeConv(in_channels, 64, 20)
        self.graph_pooling_1 = AdaptGraphPooling(4, 64, 20)
        self.gcn_2 = EdgeConv(64, 128, 20)
        self.graph_pooling_2 = AdaptGraphPooling(2, 128, 20)
        self.gcn_3 = EdgeConv(128, self.out_channel, 20)

    def forward_cls_feat(self, pos, x=None):
        if hasattr(pos, 'keys'):
            x = pos['x']
        if x is None:
            x = pos.transpose(1, 2).contiguous()

        batch_size = x.size(0)
        x1 = self.gcn_1(x) # [2B, 64, 2048]
        vertices_pool_1, x1 = self.graph_pooling_1(x, x1) # (2B,3,512), (2B,64,512)
        x2 = self.gcn_2(x1) # 2B x 128 x 512
        vertices_pool_2, x2 = self.graph_pooling_2(vertices_pool_1, x2) # (2B,3,256), (2B,128,256)
        # B x 256 x 256
        x3 = self.gcn_3(x2)

        # Global feature generating B*1024
        feat_max = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1)
        feat_avg = F.adaptive_avg_pool1d(x3, 1).view(batch_size, -1)
        feat_gf = torch.cat((feat_max, feat_avg), dim=1)

        return feat_gf
    
    def forward(self, x, features=None):
        return self.forward_cls_feat(x)