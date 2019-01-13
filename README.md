# Fast MPN-COV (i.e., iSQRT-COV)

Created by [Jiangtao Xie](http://jiangtaoxie.github.io) and [Peihua Li](http://www.peihuali.org)
<div>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="http://peihuali.org/pictures/fast_MPN-COV.JPG" width="80%"/>
</div>

## Introduction

This repository contains the source code under **MatConvNet** framework and models trained on ImageNet 2012 dataset for the following paper:

         @InProceedings{Li_2018_CVPR,
               author = {Li, Peihua and Xie, Jiangtao and Wang, Qilong and Gao, Zilin},
               title = {Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization},
               booktitle = { IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR)},
               month = {June},
               year = {2018}
         }

In this paper, we propose a fast MPN-COV method for computing matrix square root normalization, which is very efficient, scalable to multiple-GPU configuration, while enjoying matching performance with [MPN-COV](https://github.com/jiangtaoxie/MPN-COV). You can visit our [project page](http://www.peihuali.org/iSQRT-COV) for more details.

## Implementation details

We developed our programs based on MatConvNet and Matlab 2017b, running under either Ubuntu 14.04.5 LTS. To implement Fast MPN-COV meta-layer, we designed a loop-embedded directed graph, which can be divided into 3 sublayers, including `Post-normalization`, `Newton-Schulz iteration` and `Post-compensation`. Both the forward and backward propagations are performed using C++ on GPU.

## Classification Results


### Classification results (single crop 224x224, %) on ImageNet 2012 validation set

<table>
    <tr>
        <th rowspan="2" style="text-align:center;">Network</th>
        <th rowspan="2" style="text-align:center;">Top-1 Error</th>
        <th rowspan="2" style="text-align:center;">Top-5 Error</th>
        <th colspan="2" style="text-align:center;">Pre-trained models</th>
    </tr>
    <tr>
        <td style="text-align:center;">GoogleDrive</td>
        <td style="text-align:center;">BaiduCloud</td>
    </tr>
    <tr>
        <td style="text-align:center">fast MPN-COV-ResNet50</td>
        <td style="text-align:center;">22.14</td>
        <td style="text-align:center;">6.22</td>
        <td style="text-align:center;"><a href="https://drive.google.com/open?id=1fG5Mz6GzlMt7TeWq_HAr7NVqetVpgrRS">202.7MB</a></td>
        <td style="text-align:center;"><a href="https://pan.baidu.com/s/1I1XvWfx8JGB02OUHCxXpEg">202.7MB</a></td>
    </tr>
    <tr>
        <td style="text-align:center">fast MPN-COV-ResNet101</td>
        <td style="text-align:center;">21.21</td>
        <td style="text-align:center;">5.68</td>
        <td style="text-align:center;"><a href="https://drive.google.com/open?id=1ezNfxAcZNuWChIkjjC1eabVdNuVwObbS">270.5MB</a></td>
        <td style="text-align:center;"><a href="https://pan.baidu.com/s/1YuETiWAfw-RGN0sVxDlU8g">270.5MB</a></td>
    </tr>
</table>

#### Fine-grained classification results (top-1 accuracy rates, %)
<table>
     <tr>
         <th style="text-align:center;">Backbone model</th>
         <th  style="text-align:center;">Dim.</th>
         <th style="text-align:center;"><a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">Birds</a></th>
         <th  style="text-align:center;"><a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">Aircrafts</a></th>
         <th style="text-align:center;"><a href="http://www.robots.ox.ac.uk/~vgg/data/oid/">Cars</a></th>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-50</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">88.1</td>
         <td style="text-align:center;">90.0</td>
         <td style="text-align:center;">92.8</td>
     </tr>
     <tr>
         <td style="text-align:center;">ResNet-101</td>
         <td style="text-align:center;">32K</td>
         <td style="text-align:center;">88.7</td>
         <td style="text-align:center;">91.4</td>
         <td style="text-align:center;">93.3</td>
     </tr>
</table>

- Our experiments in paper are running under MatConvNet framework.
- Our method uses neither bounding boxes nor part annotations.
- We implement our source code on PyTorch toolkit, which achieve slightly better performance than MatConvNet. For more details, please refer to [PyTorch version of Fast MPN-COV](https://github.com/jiangtaoxie/fast-MPN-COV).

### Created and Modified

1. Files we created to implement fast MPN-COV meta-layer

```
└── matconvnet_root_dir
    └── matlab
        ├── +dagnn
        │   ├── OBJ_ConvNet_Cov_FroNorm.m
        │   ├── OBJ_ConvNet_COV_Pool.m
        │   ├── OBJ_ConvNet_COV_ScaleFro.m
        │   ├── OBJ_ConvNet_COV_ScaleTr.m
        │   ├── OBJ_ConvNet_Cov_Sqrtm.m
        │   └── OBJ_ConvNet_Cov_TraceNorm.m
        └── src
            ├── bits
            │   ├── impl
            │   │   ├── blashelper_cpu.hpp
            │   │   ├── blashelper_gpu.hpp
            │   │   ├── cov_froNorm_cpu.cpp
            │   │   ├── cov_froNorm_gpu.cu
            │   │   ├── cov_pool_cpu.cpp
            │   │   ├── cov_pool_gpu.cu
            │   │   ├── cov_sqrtm_cpu.cpp
            │   │   ├── cov_sqrtm_gpu.cu
            │   │   ├── cov_traceNorm_cpu.cpp
            │   │   ├── cov_traceNorm_gpu.cu
            │   │   ├── nncov_froNorm_blas.hpp
            │   │   ├── nncov_pool_blas.hpp
            │   │   ├── nncov_sqrtm_blas.hpp
            │   │   └── nncov_traceNorm_blas.hpp
            │   ├── nncov_froNorm.cpp
            │   ├── nncov_froNorm.cu
            │   ├── nncov_froNorm.hpp
            │   ├── nncov_pool.cpp
            │   ├── nncov_pool.cu
            │   ├── nncov_pool.hpp
            │   ├── nncov_sqrtm.cpp
            │   ├── nncov_sqrtm.cu
            │   ├── nncov_sqrtm.hpp
            │   ├── nncov_traceNorm.cpp
            │   ├── nncov_traceNorm.cu
            │   └── nncov_traceNorm.hpp
            ├── vl_nncov_froNorm.cpp
            ├── vl_nncov_froNorm.cu
            ├── vl_nncov_pool.cpp
            ├── vl_nncov_pool.cu
            ├── vl_nncov_sqrtm.cpp
            ├── vl_nncov_sqrtm.cu
            ├── vl_nncov_traceNorm.cpp
            └── vl_nncov_traceNorm.cu

```
2. Files we modified to support Fast MPN-COV meta-layer

```
└── matconvnet_root_dir
    └── matlab
        ├── vl_compilenn.m
        └── simplenn
            └── vl_simplenn.m
```

## Installation
1. We package our programs and [demos](./examples/imagenet) in MatConvNet toolkit,you can download this [PACKAGE](https://github.com/jiangtaoxie/fast-MPN-COV/archive/master.zip) directly, or in your Terminal type:

```Ubuntu
   >> git clone https://github.com/jiangtaoxie/fast-MPN-COV
```

2. Then you can follow the tutorial of MatConvNet's [installation guide](http://www.vlfeat.org/matconvnet/install/) to complile, for example:

```matlab
   >> vl_compilenn('enableGpu', true, ...
                   'cudaRoot', '/Developer/NVIDIA/CUDA-8.0', ...
                   'cudaMethod', 'nvcc', ...
                   'enableCudnn', true, ...
                   'cudnnRoot', 'local/cudnn-rc4') ;
```
3. Currently, we use MatConvNet 1.0-beta22. For newer versions, please consult the MatConvNet [website](http://www.vlfeat.org/matconvnet).

## Usage
### Insert MPN-COV layer into your network

1. Under SimpleNN Framework

   (1). Using traceNorm
```matlab
net.layers{end+1} = struct('type','cov_pool','name','iter_cov_pool');
net.layers{end+1} = struct('type','cov_traceNorm','name','iter_cov_traceNorm');
net.layers{end+1} = struct('type','cov_sqrtm','name','iter_cov_sqrtm','coef',1,'iterNum',5);
net.layers{end+1} = struct('type','cov_traceNorm_aux','name','iter_cov_traceNorm_aux');
```
(2). Using frobeniusNorm
```matlab
net.layers{end+1} = struct('type','cov_pool','name','iter_cov_pool');
net.layers{end+1} = struct('type','cov_froNorm','name','iter_cov_froNorm');
net.layers{end+1} = struct('type','cov_sqrtm','name','iter_cov_sqrtm','coef',1,'iterNum',5);
net.layers{end+1} = struct('type','cov_froNorm_aux','name','iter_cov_froNorm_aux');
```
2. Under DagNN Framework

   (1). Using traceNorm
```matlab
name = 'cov_pool'; % Global Covariance Pooling Layer
net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(), lastAdded.var,   name) ;
lastAdded.var = name;
name = 'cov_trace_norm'; % pre-normalization Layer by trace-Norm
name_tr =  [name '_tr'];
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_TraceNorm(),   lastAdded.var,   {name, name_tr}) ;
lastAdded.var = name;
name = 'cov_Sqrtm'; % Newton-Schulz iteration Layer
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 5),    lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
lastAdded.var = name;
lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
name = 'cov_ScaleTr'; % post-compensation Layer by trace-Norm
net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleTr(),       {lastAdded.var, name_tr},  name) ;
lastAdded.var = name;
```
   (2). Using frobeniusNorm
```matlab
name = 'cov_pool'; % Global Covariance Pooling Layer
net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(), lastAdded.var,   name) ;
lastAdded.var = name;
name = 'cov_fro_norm'; % pre-normalization Layer by frobenius-Norm
name_fro =  [name '_fro'];
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_FroNorm(),   lastAdded.var,   {name, name_fro}) ;
lastAdded.var = name;
name = 'cov_Sqrtm'; % Newton-Schulz iteration Layer
net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 5),    lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
lastAdded.var = name;
lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
name = 'cov_ScaleFro'; % post-compensation Layer by frobenius-Norm
net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleFro(),       {lastAdded.var, name_fro},  name) ;
lastAdded.var = name;
```

In our [demo](https://github.com/jiangtaoxie/demo/tree/master/imagenet) code, we implement MPN-COV AlexNet, VGG-M and VGG-VD under SimpleNN framework, and MPN-COV ResNet under DagNN framework.

###  Arguments descriptions

1. **`'coef'`**: It is reserved for future use. Currently, it should be set to 1.
2. **`'iterNum'`**: The number of Newton-Schulz iteration, 3 to 5 times is enough.

## Other Implementations

1. [PyTorch Implementation](https://github.com/jiangtaoxie/fast-MPN-COV)
2. [TensorFlow Implemention](./TensorFlow)(coming soon)

**If you have any questions or suggestions, please contact me**

`jiangtaoxie@mail.dlut.edu.cn`
