## Enriching Knowledge Distillation with Cross-Modal Teacher Fusion (Findings of CVPR)
[![arXiv](https://img.shields.io/badge/arXiv-1511.09286-<COLOR>.svg)](https://arxiv.org/abs/2511.09286)

The source code of [(Enriching Knowledge Distillation with Cross-Modal Teacher Fusion)](https://arxiv.org/abs/2511.09286).
 
 Also, see our other works:
 - [Attention as Geometric Transformation: Revisiting Feature Distillation for Semantic Segmentation](https://github.com/AmirMansurian/AttnFD) [WACV'26]
 - [A Comprehensive Survey on Knowledge Distillation](https://github.com/IPL-sharif/KD_Survey) [TMLR'25]
 - [Adaptive Inter-Class Similarity Distillation for Semantic Segmentation](https://github.com/AmirMansurian/AICSD) [MTAP'25]

<p align="left">
 <img src="https://raw.githubusercontent.com/IPL-sharif/RichKD/refs/heads/main/Figures/RichKD.png"  width="700" height="450"/>
</p>

### Installation

Environments:
- Python 3.8
- PyTorch 1.7.0

Install the package:
  ```shell
sudo pip3 install -r requirements.txt
sudo python setup.py develop
pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex tqdm
  ```

### Distillation Training on CIFAR-100
- Download the CIFAR-100 dataset and put it in ```./data```.
- Download the [cifar_teachers.tar](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and untar it to ```./download_ckpts``` via ```tar xvf cifar_teachers.tar```.
- Cache the features and logits of the CLIP using ```cache.py``` file. It saves the features and logits in ```./clip_cache```.


After doing above steps, train the student using:

  ```shell
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml

  # RichKD (L)
  python tools/train.py --cfg configs/cifar100/richkd/richkd_L.yaml

  # RichKD (F)
  python tools/train.py --cfg configs/cifar100/richkd/richkd_F.yaml

  # RichKD (L+F)
  python tools/train.py --cfg configs/cifar100/richkd/richkd.yaml
  ```
 
 ## Citation
If you use this repository for your research or wish to refer to our distillation method, please use the following BibTeX entries:
```bibtex

@inproceedings{mansourian2026enriching,
  title={Enriching knowledge distillation with cross-modal teacher fusion},
  author={Mansourian, Amir M and Babaei, Amir Mohammad and Kasaei, Shohreh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2893--2903},
  year={2026}
}


@inproceedings{mansourian2026attention,
  title={Attention as Geometric Transformation: Revisiting Feature Distillation for Semantic Segmentation},
  author={Mansourian, Amirmohammad and Jalali, Arya and Ahmadi, Rozhan and Kasaei, Shohreh},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1287--1297},
  year={2026}
}

@article{mansourian2025a,
title={A Comprehensive Survey on Knowledge Distillation},
author={Amir M. Mansourian and Rozhan Ahmadi and Masoud Ghafouri and Amir Mohammad Babaei and Elaheh Badali Golezani and Zeynab yasamani ghamchi and Vida Ramezanian and Alireza Taherian and Kimia Dinashi and Amirali Miri and Shohreh Kasaei},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025}
}

@article{mansourian2025aicsd,
  title={AICSD: Adaptive inter-class similarity distillation for semantic segmentation},
  author={Mansourian, Amir M and Ahamdi, Rozhan and Kasaei, Shohreh},
  journal={Multimedia Tools and Applications},
  pages={1--20},
  year={2025},
  publisher={Springer}
}
```

### Acknowledgement
This codebase is heavily borrowed from [Logit Standardization in Knowledge Distillation](https://github.com/sunshangquan/logit-standardization-KD). Thanks for their excellent work.
