# mlxDeepDanbooru

Pure MLX implementation of DeepDanbooru Neural Network for __Apple Silicon Chips__: M1, M2, M3, M4; 
`mlxDeepDanBooru` is available for: MacBook Pro / Air, Mac mini, iMac.

## Usage

Image-to-Text, captioning, CLIP by using [DeepDanBooru Model](https://github.com/KichangKim/DeepDanbooru) on Apple Devices.

## MLX DeepDanBooru Model 

This MLX DeepDanBooru Model implementation is inspired by a PyTorch implementation of [AUTOMATIC1111/TorchDeepDanbooru](https://github.com/AUTOMATIC1111/TorchDeepDanbooru)


## Installation

MLX is available on [PyPI](https://pypi.org/project/mlx/). To install the Python API, run:

**With `pip`**:

```
pip install mlx
```

**Clone this repo**:

```
git clone https://github.com/harryzhu/mlxDeepDanbooru.git
```

**Download MLX models and tags**:

 1) Go to `https://huggingface.co/hazhu/mlxDeepDanbooru` then 
 2) explore `models` folder,
 3) download `model-resnet_custom_v3_mlx.npz` and `tags-resnet_custom_v3_mlx.npy`
 4) and put them into `models` folder.

## Inference

```
python infer.py
```


## Performance

In the `example` folder, 1024x1024 pixl, 

On Mac Mini M4, `MLX DeepDanBooru Model` inference Speed:

```
1.7 seconds per image
```

On Mac Mini M4, __MPS + Pytorch__ inference Speed: `0.8 seconds per image` 

On Mac Mini M4, CPU + Pytorch inference Speed: `2.5 seconds per image`

## CURRENTLY  

the speed of __MPS + Pytorch__ > MLX.


## Bench: 351 images, 720x1280 and 540x720:

In Windows 11, Nvidia RTX 4070 Ti, CUDA+Pytorch:

```
SPEED: 0.3 seconds per image
Power Consumption: 260 ~ 300 Watt
```

In Mac mini M4, `mlxDeepDanBooru`:

```
SPEED: 1.68 seconds per image 
Power Consumption: 8 ~ 12 Watt
```







