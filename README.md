<div align="center">
<h1>Neighbor-Vote: Improving Monocular 3D Object Detection through Neighbor Distance Voting (ACM MM 2021)</h1>

Xiaomeng Chu, Jiajun Deng, Yao Li, Zhenxun Yuan, Yanyong Zhang, Jianmin Ji, Yu Zhang

<a href="https://arxiv.org/abs/2107.02493"><img src="https://img.shields.io/badge/arXiv-2107.02493-b31b1b" alt="arXiv"></a>
<a href="https://drive.google.com/file/d/1HszwZaMgBiJStEuw0IBUumgcDe4HOzEW/view?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Checkpoint-Orange" alt="checkpoint"></a>
</div>

```bibtex
@inproceedings{chu2021neighbor,
  title={Neighbor-vote: Improving monocular 3d object detection through neighbor distance voting},
  author={Chu, Xiaomeng and Deng, Jiajun and Li, Yao and Yuan, Zhenxun and Zhang, Yanyong and Ji, Jianmin and Zhang, Yu},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={5239--5247},
  year={2021}
}
```

## Overview

This repository is an official implementation of [Neighbor-Vote](https://dl.acm.org/doi/abs/10.1145/3474085.3475641),  a novel method that incorporates neighbor predictions to ameliorate object detection from severely deformed pseudo-LiDAR point clouds.
<div style="text-align: center;">
    <img src="docs/nv_arch.jpg" alt="Dialogue_Teaser" width=100% >
</div>


## Installation

1. Prepare for the running environment.
   
   You can follow the installation steps in [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet).
2. Prepare for the data.
   
   Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and you need to prepare your depth maps and put them to `data/kitti/training/dorn`. To provide ease of use, [PatchNet](https://github.com/xinzhuma/patchnet) provides the estimated [depth maps](https://drive.google.com/file/d/1VLG8DbjBnyLjo2OHmrb3-usiBLDcH7JF/view) generated from the pretrained models [DORN](https://github.com/hufu6371/DORN). And you can directly download the results of 2D detector FCOS on the KITTI train set from [here](https://drive.google.com/file/d/1_h9yDtHa99hh-vZjCx57u9W4Z4LOCnDv/view?usp=sharing). Please organize the downloaded files as follows:
   
   ```
   Neighbor-Vote
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2 & dorn & 2d_score_fcos
   ├── pcdet
   ├── tools
   ```
   
   Generate the data infos by running the following command:
   
   ```
   python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
   ```
3. Setup.
   
   ```
   python setup.py develop
   ```

## Model Weights

The model weights can be downloaded from [here](https://drive.google.com/file/d/1HszwZaMgBiJStEuw0IBUumgcDe4HOzEW/view?usp=sharing).

## Evaluation

The configuration file is `pointpillar.yaml` in tools/cfgs/kitti_models, and the validation scripts is in tools/scripts. 

```
cd tools
sh scripts/dist_test.sh ${NUM_GPUS} \
--cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```


## Acknowledge

Thanks to the strong and flexible [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) codebase maintained by Shaoshuai Shi ([@sshaoshuai](http://github.com/sshaoshuai)) and Chaoxu Guo ([@Gus-Guo](https://github.com/Gus-Guo)).

## Contact

This repository is implemented by Xiaomeng Chu (cxmeng@mail.ustc.edu.cn).
