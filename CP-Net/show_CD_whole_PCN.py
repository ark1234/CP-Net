#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
sys.path.append('/home/ark/local/trainningcode/PF-Net-Point-Fractal-Network-try602/')
import numpy as np
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss_test,PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD,_netG
# from model_PCN1 import AutoEncoder
# from model_PCN2 import AutoEncoder
from model_PCN3 import AutoEncoder

sys.path.append('/home/ark/local/trainningcode/PF-Net-Point-Fractal-Network-try602-test/ChamferDistancePytorch')
from chamfer3D import dist_chamfer_3D
from fscore import fscore
# sys.path.append('/home/ark/local/trainningcode/PF-Net-Point-Fractal-Network-try602/emd')
# import emd_module as emd

sys.path.append('/home/ark/local/trainningcode/PF-Net-Point-Fractal-Network-try602/emd')
import emd_module as emd

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='/home/ark/local/PF-Net-Point-Fractal-Network-datasets/dataset/shapenet_part', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='/home/ark/local/PF-Net-Point-Fractal-Network-datasets/Trained_Model_final1_Settled_Seed/point_netG80.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 


def calc_f1(output, gt, calc_f1=True):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t

def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, _ = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out
# def emd(pc1, pc2):
#     return torch.mean(earth_mover_distance(pc1, pc2)/pc1.shape[2])
# def calc_emd(output, gt, eps=0.005, iterations=50):
#     emd_loss = emd.emdModule()
#     dis, _ = emd_loss(output, gt, eps, iterations)
#     emd_out = np.sqrt(dis.cpu().detach()).mean()
#     # emd_out = np.sqrt(dis).mean()
#     return emd_out
#new
# test_dset = shapenet_part_loader.PartDataset( root='/home/ark/local/PF-Net-Point-Fractal-Network-datasets/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Airplane', npoints=opt.pnum, split='test')
test_dset = shapenet_part_loader.PartDataset( root='/home/ark/local/PF-Net-Point-Fractal-Network-datasets/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.pnum, split='test')
#new

test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=False,num_workers = int(opt.workers))
length = len(test_dataloader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num) 
# point_netG = torch.nn.DataParallel(point_netG)
# point_netG.to(device)
# point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
# point_netG.eval()

cd_loss = PointLoss().to(device)
network = AutoEncoder()
network.load_state_dict(torch.load('/home/ark/local/PF-Net-Point-Fractal-Network-datasets/PCN-datasets/100.pth'), strict=False)
network.to(device)

# testing: evaluate the mean cd loss
network.eval()

criterion_PointLoss = PointLoss_test().to(device)

input_cropped1 = torch.FloatTensor(opt.batchSize, 1, opt.pnum, 3)
errG_min = 100
n = 0
CD = 0
Gt_Pre =0
Pre_Gt = 0
IDX = 1
f1_avg= 0
emd_avg=0
for i, data in enumerate(test_dataloader, 0):
        
    real_point, target = data
    real_point = torch.unsqueeze(real_point, 1)
    batch_size = real_point.size()[0]
    real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
    input_cropped_partial =torch.FloatTensor(batch_size, 1, opt.pnum-opt.crop_point_num, 3)     
    input_cropped1.resize_(real_point.size()).copy_(real_point)
    p_origin = [0,0,0]

    choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
    index = choice[IDX-1]
    IDX  = IDX+1
    if IDX%5 == 0:
        IDX = 0
    distance_list = []
#    p_center  = real_point[0,0,index]
    p_center = index
    for num in range(opt.pnum):
        distance_list.append(distance_squre(real_point[0,0,num],p_center))
    distance_order = sorted(enumerate(distance_list), key = lambda x:x[1])
    
    for sp in range(opt.crop_point_num):
        input_cropped1.data[0,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
        real_center.data[0,0,sp] = real_point[0,0,distance_order[sp][0]]
        
    crop_num_list = []
    for num_ in range(opt.pnum-opt.crop_point_num):
        crop_num_list.append(distance_order[num_+opt.crop_point_num][0])
    indices = torch.LongTensor(crop_num_list)
    input_cropped_partial[0,0]=torch.index_select(real_point[0,0],0,indices)
    input_cropped_partial = torch.squeeze(input_cropped_partial,1)
    input_cropped_partial = input_cropped_partial.to(device)
     
    real_center = torch.squeeze(real_center,1)
#    real_center_key_idx = utils.farthest_point_sample(real_center,64,train = False)
#    real_center_key = utils.index_points(real_center,real_center_key_idx)
#    input_cropped1 = input_cropped1.to(device)
    
    input_cropped1 = torch.squeeze(input_cropped1,1)
    # print(input_cropped1.shape)
    # exit()
    # input_cropped2_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    # input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
    # input_cropped3_idx = utils.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
    # input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
    # input_cropped2 = input_cropped2.to(device)
    # input_cropped3 = input_cropped3.to(device)      
    # input_cropped  = [input_cropped1,input_cropped2,input_cropped3]
    
#    fake,fake_part = point_netG(input_cropped)
    # fake_center1,fake_center2,fake=point_netG(input_cropped1)
    # v, y_coarse, y_detail = network(partial_input)
    # print(input_cropped1.shape)
    # exit()
    input_cropped1 = input_cropped1.to(device)
    input_cropped1 = input_cropped1.permute(0, 2, 1)
    fake_center1,fake_center2,fake = network(input_cropped1)
    fake_whole = fake.permute(0, 2, 1) #torch.Size([1, 16384, 3])
    # fake_idx = utils.farthest_point_sample(fake,2048,RAN = False)
    # fake_whole     = utils.index_points(fake,fake_idx )


    fake = fake.permute(0, 2, 1)
    # fake_center1,fake_center2,fake=point_netG(input_cropped)
    fake_whole = torch.cat((input_cropped_partial,fake),1)
    # print(fake_whole.shape)
    # print(fake.shape)
    # exit()
    fake_whole = fake_whole.to(device)
    real_point = real_point.to(device)
    real_center = real_center.to(device)


    #new
    # dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(real_center,1))#+0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
    # dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake_whole,1),torch.squeeze(real_point,1))#+0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
    #new
    dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake_whole,1),torch.squeeze(real_point,1))
    
    # _, _,f1 = calc_f1(torch.squeeze(fake_whole,1),torch.squeeze(real_point,1))
    _, _,f1 = calc_f1(torch.squeeze(fake_whole,1),torch.squeeze(real_point,1))
    f1=f1.cpu().detach().numpy()
    # print(fake_whole.shape)
    # exit()
    # f1_avg=f1.mean().item()
    # emd0= earth_mover_distance(fake_whole, real_point, transpose=True)
    emd0 = calc_emd(torch.squeeze(fake_whole,1), torch.squeeze(real_point,1))
    # # emd0 = calc_emd(fake_whole, real_point)
    emd0=emd0.cpu().detach().numpy()
    # print(f1)
    # print(emd0)
    # exit()


    dist_all=dist_all.cpu().detach().numpy()
    dist1 =dist1.cpu().detach().numpy()
    dist2 = dist2.cpu().detach().numpy()
    CD = CD + dist_all/length
    Gt_Pre = Gt_Pre + dist1/length
    Pre_Gt = Pre_Gt + dist2/length

    f1_avg = f1_avg + f1/length
    emd_avg = emd_avg + emd0/length
    print(CD,Gt_Pre,Pre_Gt,f1_avg,emd_avg)
print(CD,Gt_Pre,Pre_Gt,f1_avg,emd_avg)
print("CD:{} , Gt_Pre:{} , Pre_Gt:{}, f1_avg:{}, emd_avg:{}".format(float(CD),float(Gt_Pre),float(Pre_Gt),float(f1_avg),float(emd_avg)))
print(length)    
    
    
