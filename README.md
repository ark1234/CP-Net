# CP-Net
Cosmos Propagation Network: Deep learning model for point cloud completion

This repository is still under constructions.

If you have any questions about the code, please contact me. Thanks!


##1) Dataset
###1) PF-Net dataset
For PF-Net dataset (CVPR2020 PF-Net: Point Fractal Network for 3D Point Cloud Completion)
```
  cd dataset
  bash download_shapenet_part16_catagories.sh
  You can also download the dataset from 
  链接：https://pan.baidu.com/s/1MavAO_GHa0a6BZh4Oaogug 提取码：3hoe 
```
##2) Train
```
python Train_FPNet.py 
```
Change ‘crop_point_num’ to control the number of missing points.
Change ‘point_scales_list ’to control different input resolutions.
Change ‘D_choose’to control without using D-net.

##3) Evaluate the Performance on ShapeNet
```
python show_recon.py
```
Show the completion results, the program will generate txt files in 'test-examples'.
```
python show_CD.py
```
Show the Chamfer Distances and two metrics in our paper.

##4) Visualization of csv File

We provide some incomplete point cloud in file 'test_one'. Use the following code to complete a incomplete point cloud of csv file:
```
python Test_csv.py
```
change ‘infile’and  ‘infile_real’to select different incomplete point cloud in ‘test_one’

##5) Visualization of Examples

Using Meshlab to visualize  the txt files.
