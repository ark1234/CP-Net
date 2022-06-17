# CP-Net
Cosmos Propagation Network: Deep learning model for point cloud completion

This repository is still under constructions.

If you have any questions about the code, please contact me. Thanks!


### Usage

#### 1) Envrionment & prerequisites

- Pytorch 1.2.0
- CUDA 10.0
- Python 3.7
- [Visdom](https://github.com/facebookresearch/visdom)
- [Open3D](http://www.open3d.org/docs/release/index.html#python-api-index)

#### 2) Compile

Compile extension modules from [msn](https://github.com/Colin97/MSN-Point-Cloud-Completion) for Evaluate the Performance with EMD and F1 socre (No need if only use Chamfer Distancesm):  

    git clone https://github.com/Colin97/MSN-Point-Cloud-Completion
    cd emd
    python3 setup.py install
    cd expansion_penalty
    python3 setup.py install
    cd MDS
    python3 setup.py install

### ShapenetPart dataset 
```
  cd dataset
  bash download_shapenet_part16_catagories.sh
  You can also download the dataset from 
  链接：https://pan.baidu.com/s/1MavAO_GHa0a6BZh4Oaogug 提取码：3hoe 
```

### [ShapeNet](https://www.shapenet.org/), [Compeletion3D](http://completion3d.stanford.edu/) are available below:

- [ShapeNet](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
- [Completion3D](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip)

### Train
```
python Train_FPNet.py 
```
Change ‘crop_point_num’ to control the number of missing points.
Change ‘D_choose’to control without using D-net.

### Evaluate the Performance on ShapeNet and other datasets
```
python show_recon.py
```
Show the completion results, the program will generate txt files in 'test-examples'.
```
python show_CD.py
```
Show the Chamfer Distancesm, EMD and F1.

### Visualization of csv File

We provide some incomplete point cloud in file 'test_one'. Use the following code to complete a incomplete point cloud of csv file:
```
python Test_csv.py
```
change ‘infile’and  ‘infile_real’to select different incomplete point cloud in ‘test_one’

### Visualization of Examples

Using Meshlab or Cloudcompare to visualize the txt/csv files.
