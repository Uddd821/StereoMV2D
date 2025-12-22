# StereoMV2D
This repo is the official PyTorch implementation for paper:   
[StereoMV2D: A Sparse Temporal Stereo-Enhanced Framework for Robust Multi-View 3D Object Detection]. 

We propose StereoMV2D, a unified framework that integrates temporal stereo modeling into the 2D detection–guided multiview 3D detector. By exploiting cross-temporal disparities of the same object across adjacent frames, StereoMV2D enhances depth perception and refines the query priors, while performing all computations efficiently within 2D regions of interest (RoIs). Furthermore, a dynamic confidence gating mechanism adaptively evaluates the reliability of temporal stereo cues through learning statistical patterns derived from the inter-frame matching matrix together with appearance consistency, ensuring robust detection under object appearance and occlusion.
## Preparation
This implementation is built upon [MV2D](https://github.com/tusen-ai/MV2D), and can be constructed as the [install.md](https://github.com/megvii-research/PETR/blob/main/install.md).

* Environments  
  Linux, Python == 3.8.10, CUDA == 11.3, pytorch == 1.11.0, mmcv == 1.6.1, mmdet == 2.25.1, mmdet3d == 1.0.0, mmsegmentation == 0.28.0   

* Detection Data   
Follow the mmdet3d to process the nuScenes dataset (https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/data_preparation.md).

* Pretrained weights   
We use nuImages pretrained weights from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/nuimages). Download the pretrained weights and put them into `weights/` directory. 

* After preparation, you will be able to see the following directory structure:  
  ```
  StereoMV2D
  ├── mmdetection3d
  ├── configs
  ├── mmdet3d_plugin
  ├── tools
  ├── data
  │   ├── nuscenes
  │     ├── ...
  ├── weights
  ├── README.md
  ```

## Train & Inference
<!-- ```bash
git clone https://github.com/Uddd821/StereoMV2D.git
``` -->
```bash
cd StereoMV2D
```
You can train the model following:
```bash
bash tools/dist_train.sh configs/StereoMV2d/exp/stream_r50_frcnn_1408x512_ep24.py 8 
```
You can evaluate the model following:
```bash
bash tools/dist_test.sh configs/StereoMV2d/exp/stream_r50_frcnn_1408x512_ep24.py work_dirs/stream_r50_frcnn_1408x512_ep24/latest.pth 8 --eval bbox
```

## Acknowledgement
Many thanks to the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [petr](https://github.com/megvii-research/PETR/tree/main) and [TOC3D](https://github.com/DYZhang09/ToC3D).
