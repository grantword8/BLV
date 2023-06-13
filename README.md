## Balancing Logit Variation for Long-tailed Semantic Segmentation, CVPR 2023

- [x] Fully-supervised semantic segmentation.
- [ ] Unsupervised Domain adaptive semantic segmentation.
- [ ] Semi-supervised semantic segmentation.

## Fully-supervised semantic segmentation


### Installation
```shell
cd fully_sup
conda create -n blv python=3.7 -y
conda activate blv
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install openmim
mim install mmcv-full==1.4.0
pip install -e .
```


### Data Preparation
Please follow this link [dataset_prepare.md](https://github.com/open-mmlab/mmsegmentation/blob/v0.23.0/docs/en/dataset_prepare.md#prepare-datasets) to setup the datasets.



### Run
For some models, you should download the corresponding pretrained checkpoints for the backbone manually.
```shell
cd fully_sup
python -u tools/train.py /path/to/the/config/file
```



### Results and configs
| Method    | Backbone  | mIoU | mIoU(tail) | config |
| ---- | ---- | ---- | ---- | ---- |
|HRNet-18| OCRHead|79.22|63.51|[config](./fully_sup/configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes.py)|
|HRNet-18| +BLV|79.94|66.70|[config](./fully_sup/configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes_blv.py)|
|ResNet50| UperHead|78.28|62.56|[config](./fully_sup/configs/upernet/upernet_r50_512x1024_40k_cityscapes.py)|
|ResNet50| +BLV|78.63|64.57|[config](./fully_sup/configs/upernet/upernet_r50_512x1024_40k_cityscapes_blv.py)|
|ResNet50| PSPHead|77.98|61.96|[config](./fully_sup/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py)|
|ResNet50| +BLV|78.53|63.34|[config](./fully_sup/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes_blv.py)|
|ResNet101| UperHead|79.41|64.68|[config](./fully_sup/configs/upernet/upernet_r101_512x1024_40k_cityscapes.py)|
|ResNet101| +BLV|79.88|66.29|[config](./fully_sup/configs/upernet/upernet_r101_512x1024_40k_cityscapes_blv.py)|
|MiT-b0| SegformerHead|76.85|67.58|[config](./fully_sup/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py)|
|MiT-b0| +BLV|77.09|68.91|[config](./fully_sup/configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_blv.py)|
|Swin-T| K-NeT|79.68|71.70|[config](./fully_sup/configs/knet/knet_s3_upernet_swin-t_8x2_512x1024_adamw_80k_cityscapes.py)|
|Swin-T| +BLV|80.11|72.94|[config](./fully_sup/configs/knet/knet_s3_upernet_swin-t_8x2_512x1024_adamw_80k_cityscapes_blv.py)|
|Vit-B16| K-NeT|76.48|68.25|[config](./fully_sup/configs/vit/upernet_vit-b16_mln_512x1024_80k_cityscapes.py)|
|Vit-B16| +BLV|77.68|70.63|[config](./fully_sup/configs/vit/upernet_vit-b16_mln_512x1024_80k_cityscapes_blv.py)|

## Unsupervised Domain adaptive semantic segmentation

## Semi-supervised semantic segmentation

### Citation
If you find this useful for your research, please cite the following paper.

```latex
@inproceedings{wang2023balancing,
  title={Balancing Logit Variation for Long-tailed Semantic Segmentation},
  author={Wang, Yuchao and Fei, Jingjing and Wang, Haochen and Li, Wei and Bao, Tianpeng and Wu, Liwei and Zhao, Rui and Shen, Yujun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19561--19573},
  year={2023}
}
```

### Acknowledgements


The implementation of fully-supervised semantic segmentation task is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.23.0/mmseg).


The implementation of unsupervised domain adaptived semantic segmentation task is based on [HRDA](https://github.com/lhoyer/HRDA).