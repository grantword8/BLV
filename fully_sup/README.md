## Fully-supervised semantic segmentation

### Installation
```shell
cd 
conda create -n blv python=3.7 -y
conda activate blv
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install openmim
mim install mmcv-full==1.4.0
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
|HRNet-18| OCRHead|79.22|63.51|[config]()|
|HRNet-18| +BLV|79.94|66.70|[config]()|
|ResNet50| UperHead|78.28|62.56|[config]()|
|ResNet50| +BLV|78.63|64.57|[config]()|
|ResNet50| PSPHead|77.98|61.96|[config]()|
|ResNet50| +BLV|78.53|63.34|[config]()|
|ResNet101| UperHead|79.41|64.68|[config]()|
|ResNet101| +BLV|79.88|66.29|[config]()|
|MiT-b0| SegformerHead|76.85|67.58|[config]()|
|MiT-b0| +BLV|77.09|68.91|[config]()|
|Swin-T| K-NeT|79.68|71.70|[config]()|
|Swin-T| +BLV|80.11|72.94|[config]()|
|Vit-B16| K-NeT|76.48|68.25|[config]()|
|Vit-B16| +BLV|77.68|70.63|[config]()|
