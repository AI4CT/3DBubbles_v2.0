**Language version: [English](README.md), [中文](README_zh.md).**

# 3DBubbles

<img src="img/3DBubbles_logo.png" width="60%" style="margin: 0 auto;">

A publicly available gas-liquid two-phase flow dataset (3DBubbles) following the Apache License 2.0 protocol provides high-precision bubble flow 3D and 2D data and statistical information.

3DBubbles includes the following three types of data:
1. Single bubble 3D structures in ``stl'' format. The bubble structure is generated from a static gas-liquid flow field phantom, which is digitized and post-processed by a in-house developed high spatial resolution X-ray CT system;
2. Single bubble 2D images in ``png'' format. Obtained by rasterized rendering of the 3D point cloud, each 3D bubble provides a rendered image with 26 angles.
3. 3D bubbly flow. Includes macroscopic "stl" format 3D bubbly flow and rasterized rendered 2D bubbly flow images.

Notice:
1. The first two data have been open-sourced on the Kaggle platform at the following link: <https://www.kaggle.com/datasets/ai4ctofipe/3dbubbles>. If you need it, please give proper citation according to the last format, thanks!
2. 3D bubbly flow data is generated via [FlowRenderer.py](FlowRenderer.py) file.

## File tree of 3DBubbles dataset

    3DBubbles
    ├── parameters.csv % Structural parameters and geometric information of 3D bubble and 2D projection images
    ├── reconstruction_characterization.csv % Reconstruction indicators and structural parameters (SH degrees from 2 to 20)
    ├── mesh % 10823 bubble mesh files in ``stl'' format
    │   ├── 00001.stl
    │   ├── 00002.stl
    │   ├── ……
    │   └── 10823.stl  
    ├── projection % 10823 x 26 projected images
    │   ├── 00001
    │   │   ├── Sphere_0.00_0.00_1.00_scale=39.png  
    │   │   ├── Sphere_0.00_0.00_-1.00_scale=39.png    
    │   │   ├── ……
    │   │   └── Sphere_-1.00_0.00_0.00_scale=39.png
    │   ├── 00002
    │   ├── ……
    │   └── 10823
    └── SH_coefficient % 10823 x 20 spherical harmonic coefficients
        ├── 00001
        │   ├── N=1.npy  
        │   ├── N=2.npy
        │   ├── ……
        │   └── N=20.npy
        ├── 00002
        ├── ……
        └── 10823

## Usage
### Environment
Current version only supports training and inference on CPU. It works well under dependencies as follows:
* Ubuntu 24.04 / Windows 10 Professional
* Python 3.8 / 3.12
* numpy 1.24.4 / 1.26.0
* pyvista 0.44.1 / 0.44.1
* scipy 1.10.1 / 1.14.1
* matplotlib 3.7.5 / 3.9.3

### Data preparation
Before running the code, please download the 3DBubbles dataset at the Kaggle platform: <https://www.kaggle.com/datasets/ai4ctofipe/3dbubbles>, which has 10,823 stl files.

**Note that we uploaded only 20 of these stl files at GitHub.**

### 3D Bubble Flow Generation

Run the [FlowRenderer.py](FlowRenderer.py) file.

Example: Generate a bubbly flow and rendering of a 5mm cube with 1% gas holdup:

```
python .\FlowRenderer.py -num 1 -x 5 -y 5 -hh 5 --gas_holdup 0.01
```

Results:

[3D digital model of bubbly flow](3Dbubbleflowrender/20250110T172559-117775/000/0.01_fibonacci.stl)

<img src="img/3D_Bubbly_flow.png" width="60%">

Bubbly Flow Image：

<img src="3Dbubbleflowrender/20250110T172559-117775/000/bubbly_flow.png" width="60%">

Bubbly Flow Image With Detection Boxes：

<img src="3Dbubbleflowrender/20250110T172559-117775/000/bubbly_flow_bboxes.png" width="60%">

Bubble Masks：

<img src="3Dbubbleflowrender/20250110T172559-117775/000/mask.png" width="60%">

Bubbly Flow Image with Masks：

<img src="3Dbubbleflowrender/20250110T172559-117775/000/mask_merge.png" width="60%">

Bubbles with detection boxes and masks for bubbly flow image：

<img src="3Dbubbleflowrender/20250110T172559-117775/000/mask_merge_bboxes.png" width="60%">

The optional arguments provided by argparse are as follows:
```
usage: FlowRenderer.py [-h] [--stl_path STL_PATH] [--save_path SAVE_PATH] [-num FLOW_NUM] [-x VOLUME_SIZE_X] [-y VOLUME_SIZE_Y] [-hh VOLUME_HEIGHT] [--gas_holdup GAS_HOLDUP] [-a ALPHA] [-t TRUNCATION]
                       [--poisson_max_iter POISSON_MAX_ITER] [--sample_spacing SAMPLE_SPACING]

流场生成器与渲染器

options:
  -h, --help            show this help message and exit
  --stl_path STL_PATH   STL文件的路径
  --save_path SAVE_PATH
                        保存路径
  -num FLOW_NUM, --flow_num FLOW_NUM
                        生成数量
  -x VOLUME_SIZE_X, --volume_size_x VOLUME_SIZE_X
                        流场宽度[mm]
  -y VOLUME_SIZE_Y, --volume_size_y VOLUME_SIZE_Y
                        流场深度[mm]
  -hh VOLUME_HEIGHT, --volume_height VOLUME_HEIGHT
                        流场高度[mm]
  --gas_holdup GAS_HOLDUP
                        气含率
  -a ALPHA, --alpha ALPHA
                        向量指数:Alpha
  -t TRUNCATION, --truncation TRUNCATION
                        截断值
  --poisson_max_iter POISSON_MAX_ITER
                        泊松圆盘采样最大迭代次数
  --sample_spacing SAMPLE_SPACING
                        点云上采样的采样距离
```
## Coming soon
See [Update_log.md](Update_log.md) for the update log.

### Bubbly flow

1.  3D bubble flow field with different gas holdups of 0.5%, 1%, 2%, etc. (mesh information, STL format);

2.  bubble flow statistics (bubble size distribution BSD, bubble angle distribution BRD, bubble position distribution BPD, etc.)；

3.  More constraints (same as 2).

## Contact

Author: Baodi Yu (于宝地)，e-mail: <yubaodi20@ipe.ac.cn>
Research Group: AI4CT

## Citation
Please cite our paper if you find this code useful:
 ```
@misc{3DBubbles,
  author={Baodi Yu, Qian Chen, Yanwei Qin1, Sunyang Wang, Xiaohui Su, Fanyong Meng},
  title={3DBubbles: an experimental dataset for model training and validation},
  year={2025},
  howpublished = {\url{https://github.com/Baodi-Yu/3DBubbles}}
}
 ```