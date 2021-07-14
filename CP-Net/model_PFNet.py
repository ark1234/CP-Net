#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import os
import sys
import copy
import math
import numpy as np
import torch.nn.init as init

class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128),2)
        x_256 = torch.squeeze(self.maxpool(x_256),2)
        x_512 = torch.squeeze(self.maxpool(x_512),2)
        x_1024 = torch.squeeze(self.maxpool(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        return x

class Latentfeature0(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self,x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs,2)
        latentfeature = latentfeature.transpose(1,2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature,1)
        # print('latentfeature',latentfeature.shape)
#        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))  
#        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
#        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
#        latentfeature = latentfeature + latentfeature_64
#        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
#        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
#        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
#        latentfeature = latentfeature + latentfeature_256
#        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
#        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
#        latentfeature = self.maxpool(latentfeature)
#        latentfeature = torch.squeeze(latentfeature,2)
        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list,k=40):
        super(PointcloudCls,self).__init__()
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.latentfeature(x) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))        
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class _netG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(128,64,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(64,3,1)
        # self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(64,3,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        
    def forward(self,x):
        x = self.latentfeature(x) #[B,1920]
        # x_1 = F.relu(x)
        x_1 = F.relu(self.fc1(x)) # [B,1920] -> [B,1024]
        x_2 = F.relu(self.fc2(x_1)) # [b, 1024] -> [B,512]
        x_3 = F.relu(self.fc3(x_2))  # [b,512] -> [B,256]
        
        
        pc1_feat = self.fc3_1(x_3) # [B,64x3]
        pc1_xyz = pc1_feat.reshape(-1,3,64) # [B,64x3] -> #[b, 3, 64]
        # print(pc1_xyz.shape)
        # exit() 
        
        pc2_feat = F.relu(self.fc2_1(x_2)) # [B,512] -> [b, 64x128]
        pc2_feat = pc2_feat.reshape(-1,64,128) # [b, 64x128] -> [b, 64, 128]
        pc2_xyz =self.conv2_1(pc2_feat) #[b, 64, 128] -> [B, 3, 128]
        # print(pc2_xyz.shape)
        # exit() 
        pc3_feat = F.relu(self.fc1_1(x_1)) # [B,1024] -> [B,128*512]
        pc3_feat = pc3_feat.reshape(-1,128,512) #  [B,128*512] -> [B,128, 512]
        pc3_feat = F.relu(self.conv1_1(pc3_feat)) #  [B,128, 512] -> [B, 64, 512]
        pc3_xyz = self.conv1_2(pc3_feat) # #  [B,64, 512] -> [B,3, 512]
        
        pc_xyz_low = pc1_xyz
        pc_xyz_low = pc_xyz_low.transpose(1,2)
#new

        center_1 = torch.unsqueeze(pc1_xyz,-1) #[B, 3, 64, 1]
        # print('pc1_xyz',pc1_xyz.shape)
        mirror_1 = get_graph_feature1(pc1_xyz, k=1)  #[B, 3, 64, 1]
        # print(A.shape)
        # exit() 
        mirror_1_ = 2*center_1 - mirror_1  #[B, 3, 64, 1]
        # print(pc1_xyz_expand.shape)
        pc1_xyz_expand =torch.cat((mirror_1_ , center_1),-1)#[b, 3, 64, 2]

        pc1_xyz_expand =pc1_xyz_expand.reshape(-1,3,128) #[B, 3, 128]
 


        pc_xyz_middle = pc1_xyz_expand+pc2_xyz #[B, 3, 128]]
        pc_xyz_middle = pc_xyz_middle.transpose(1,2)


       
#new
        # pc2_xyz_expand = torch.unsqueeze(pc2_xyz,2)
        center_2 = torch.unsqueeze(pc2_xyz, -1)  #[B, 3, 128, 1]

        mirror_2 = get_graph_feature1(pc2_xyz, k=3) #[b, 3, 128, 3]
        # print('mirror_2',mirror_2.shape)
        # print(A2.shape) 
        # exit() 
        mirror_2_ = 2*center_2 - mirror_2 #A_dash #[b, 3, 128, 3]
        # print(pc2_xyz_expand.shape) 
        pc2_xyz_expand = torch.cat((mirror_2_, center_2),-1) #[b, 3, 128, 4]
        pc2_xyz_expand = pc2_xyz_expand.reshape(-1,3,512)

        # print(pc2_xyz_expand.shape) 
        # exit() 
#new

        # pc3_xyz = pc3_xyz.transpose(1,2)
        # pc3_xyz = pc3_xyz.reshape(-1,128,int(self.crop_point_num/128),3) #[40, 128, 4, 3]
        # print(pc3_xyz.shape)
        # exit() 
        pc_xyz_high = pc2_xyz_expand+pc3_xyz #[b,3,512]
        pc_xyz_high = pc_xyz_high.transpose(1,2) #[b,3,512] -> #[b,512,3]
        # pc3_xyz = pc3_xyz.reshape(-1,self.crop_point_num,3) 
        
        # return pc1_xyz,pc2_xyz,pc3_xyz #center1 ,center2 ,fine
        # print("pc_xyz_low",pc_xyz_low.shape)
        # print("pc_xyz_middle",pc_xyz_middle.shape)
        # print("pc_xyz_high",pc_xyz_high.shape)
        return pc_xyz_low, pc_xyz_middle, pc_xyz_high

class _netlocalD(nn.Module):
    def __init__(self,crop_point_num):
        super(_netlocalD,self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,16)
        self.fc4 = nn.Linear(16,1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = torch.squeeze(self.maxpool(x_64))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        Layers = [x_256,x_128,x_64]
        x = torch.cat(Layers,1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x



class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature, self).__init__()
        # self.args = args
        self.k = 16
        
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn3_1 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm1d(1024)

        # self.bn1_1 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2_1 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False),
                                   self.bn2_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn3_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(768, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv2_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
        #                            self.bn1_1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 2048, bias=False)
        self.bn6 = nn.BatchNorm1d(2048)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

#new
        # self.linear4 = nn.Linear(64*2048, 256)
        # self.linear5 = nn.Linear(64*2048, 256)
        # self.linear6 = nn.Linear(128*2048, 512)
        # self.linear7 = nn.Linear(256*2048, 1024)
        # self.linear8 = nn.Linear(256*2048, 1024)
#new

    def forward(self, x):
        batch_size = x.size(0)
        # print(x.shape)
        # exit()
        x =x.permute(0,2,1)
        x,idx0 = get_graph_feature2(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # print(x.shape)
        # exit()
        # print(x.type())
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        # print(x1.shape)
        # exit()
#new
        # x01 = x1.clone().detach()
        # x01 =x01.permute(0,2,1)
        # x01 =x01.reshape(-1,64*2048)
        # x01 = self.linear4(x01)
#new
        # x1 =x1.permute(0,2,1)
        # print(x1.shape)
        # x01 = x1.clone().detach()
        x = get_graph_feature3(x1, k=self.k, idx=idx0 )     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        # print(x.shape)
        # exit()
        # print(x.type())
        # x = x.float()
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#new
        # x02 = x2.clone().detach()
        # x02 =x02.permute(0,2,1)
        # x02 =x02.reshape(-1,64*2048)
        # x02 = self.linear5(x02)
#new
        # x2 =x2.permute(0,2,1)
        # x02 = x2.clone().detach()
        # print(x1.shape)
        # exit()
        # x2 = torch.cat((x1, x2), dim=-1)      # (batch_size, 128, num_points)
        x = torch.cat((x1, x2), dim=1)
        # print(x.shape)
        # exit()
        x = self.conv2_1(x)
        # print(x2.shape)
        # exit()
        x = get_graph_feature3(x, k=self.k, idx=idx0)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        # print(x.shape)
        # exit()
        x = self.conv3(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

#new
        # x03 = x3.clone().detach()
        # x03 =x03.permute(0,2,1)
        # x03 =x03.reshape(-1,128*2048)
        # x03 = self.linear6(x03)
#new
        # x3 =x3.permute(0,2,1)
        # x03 = x3.clone().detach()
        # print(x3.shape)
        # exit()
        # x3 = torch.cat((x1, x2, x3), dim=-1)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv3_1(x)
        # print(x3.shape)
        # exit()
        x = get_graph_feature3(x, k=self.k, idx=idx0)     # (batch_size, 256, num_points) -> (batch_size, 256*2, num_points, k)
        # print(x.shape)
        # exit()
        x = self.conv4(x)                       # (batch_size, 256*2, num_points, k) -> (batch_size, 512, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 512, num_points, k) -> (batch_size, 512, num_points)
        # x4 =x4.permute(0,2,1)
#new
        # x04 = x4.clone().detach()
        # # x04 =x04.permute(0,2,1)
        # # x04 =x04.reshape(-1,256*2048)
        # # x04 = self.linear7(x04)

        # # x4 =x4.permute(0,2,1)
        # latentfeature = torch.cat((x01, x02, x3), dim=-1)
        # latentfeature =latentfeature.reshape(-1,256*2048)
        # latentfeature = self.linear8(latentfeature)
        # # print(latentfeature.shape)
        # # exit()
        # latentfeature = latentfeature.unsqueeze(1)
        # latentfeature = F.relu(self.bn5(latentfeature))
        # # latentfeature = torch.squeeze(latentfeature,1)
        # latentfeature = torch.squeeze(latentfeature,1)
        # # print(latentfeature.shape)
        # # exit()
#new



        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # exit()
        # x = torch.cat((x1, x2, x3, x4), dim=-1)  # (batch_size, 64+128+512, num_points)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # print(x.shape)
        # exit()
        # x =x.permute(0,2,1)
        x = self.conv5(x)                       # (batch_size, 768, num_points) -> (batch_size, 1024, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        # print(x.shape)
        # exit()
        latentfeature = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        # x = self.dp2(x)
        # x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        # print(latentfeature.shape)
        # exit()
        return latentfeature


#new
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature1(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
#new


    # device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#new

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx.to(device)

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = feature.permute(0, 3, 1, 2)
  
    return feature      # (batch_size, num_dims, num_points, k)



def get_graph_feature2(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    # print(batch_size)
    # print(num_points)
    # exit() 
    x = x.view(batch_size, -1, num_points)
    # x = x.permute(0,2,1)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

#new
    idx0 = idx.clone().detach()
    # device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#new

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature, idx0      # (batch_size, 2*num_dims, num_points, k)


def get_graph_feature3(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    # print(batch_size)
    # print(num_points)
    # exit() 
    x = x.view(batch_size, -1, num_points)
    # x = x.permute(0,2,1)
    # if idx is None:
    #     idx = knn(x, k=k)   # (batch_size, num_points, k)

#new
    # device = torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#new

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)











# def knn2(x, k):
#     inner = -2*torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
#     return idx


# def get_graph_feature2(x, k, idx=None):
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         idx = knn2(x, k=k)   # (batch_size, num_points, k)
#     device = torch.device('cuda')

#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

#     idx = idx + idx_base

#     idx = idx.view(-1)
 
#     _, num_dims, _ = x.size()

#     # x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size*num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims) 
#     # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
#     feature = feature.permute(0, 3, 2, 1).contiguous()
  
#     return feature


#new


if __name__=='__main__':
    input1 = torch.randn(64,2048,3)
    input2 = torch.randn(64,512,3)
    input3 = torch.randn(64,256,3)
    input_ = [input1,input2,input3]
    # input_ = input1
    netG=_netG(3,1,[2048,512,256],1024)
    output = netG(input_)
    # print(output)
