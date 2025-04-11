# Gaussian On-The-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization
Yiwei Xu, Yifei Yu, Wentian Gan, Tengfei Wang, Zongqian Zhan, Hao Cheng and Xin Wang

[![arxiv](https://img.shields.io/badge/arxiv-2406.15643-red)](https://arxiv.org/abs/2503.13086)
[![webpage](https://img.shields.io/badge/webpage-green)](https://xywjohn.github.io/GS_On-the-Fly.github.io/)

<p align="center">
    <img src="Final_Demo.mp4" width="700px"/>
</p>

## Data Preparation
Since this project aims to enable simultaneous image acquisition and 3D Gaussian Splatting (3DGS) training, we need to utilize the On-The-Fly Structure-from-Motion (SfM) system proposed by Zhan et al. This system has already achieved the capability of near real-time image acquisition and camera pose estimation. In this project, we will leverage the camera poses and sparse point clouds provided by this system as the input for subsequent 3DGS training.

You can use your own data or the test data provided by us (located in /demo_data/images) to perform processing with the On-The-Fly SfM system. This will produce results as illustrated below:

│  
├─16
│  ├─images
│  │      1DJI_0024.JPG
│  │      1DJI_0025.JPG
│  │      ......
│  │      
│  └─sparse
│      └─0
│              cameras.bin
│              imageMatchMatrix.txt
│              images.bin
│              imagesNames.txt
│              points3D.bin
│              points3D.ply
│              
├─17
│  ├─images
│  │      1DJI_0023.JPG
│  │      1DJI_0024.JPG
│  │      ......
│  │      
│  └─sparse
│      └─0
│              cameras.bin
│              imageMatchMatrix.txt
│              images.bin
│              imagesNames.txt
│              points3D.bin
│              points3D.ply
│              
├─18
│  ......
├─19
│  ......
├─20
│  ......
├─21
│  ......

## BibTeX
```
@misc{xu2025gaussianontheflysplattingprogressive,
      title={Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization}, 
      author={Yiwei Xu and Yifei Yu and Wentian Gan and Tengfei Wang and Zongqian Zhan and Hao Cheng and Xin Wang},
      year={2025},
      eprint={2503.13086},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.13086}, 
}
```