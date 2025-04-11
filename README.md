# Gaussian On-The-Fly Splatting
# A Progressive Framework for Robust Near Real-Time 3DGS Optimization
Yiwei Xu, Yifei Yu, Wentian Gan, Tengfei Wang, Zongqian Zhan, Hao Cheng and Xin Wang

**Abstract:** 3D Gaussian Splatting (3DGS) achieves high fidelity rendering with fast real-time performance, but existing methods rely on offline training after full Structure-from Motion (SfM) processing. In contrast, this work introduces On the-Fly GS, a progressive framework enabling near real-time 3DGS optimization during image capture. As each image arrives, its pose and sparse points are updated via on-the-fly SfM, and newly optimized Gaussians are immediately integrated into the 3DGS field. We propose a progressive local optimization strategy to prioritize new images and their neighbors by their corresponding overlapping relationship, allowing the new image and its overlapping images to get more training. To further stabilize training across old and new images, an adaptive learning rate schedule balances the iterations and the learning rate. Moreover, to maintain overall quality of the 3DGS field, an efficient global optimization scheme prevents overfitting to the newly added images. Experiments on multiple benchmark datasets show that our On-the-Fly GS reduces training time significantly, optimizing each new image in seconds with minimal rendering loss, offering the first practical step toward rapid, progressive 3DGS reconstruction.

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

---

