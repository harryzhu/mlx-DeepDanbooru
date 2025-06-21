# mlxDeepDanbooru
Pure MLX implementation of DeepDanbooru neural network for Apple Silicon Chips: M1, M2, M3, M4

## Usage

Image-to-Text, CLIP by using [DeepDanBooru Model](https://github.com/KichangKim/DeepDanbooru)

## MLX Model 

This MLX DeepDanBooru Model implementation is from a PyTorch implementation of [AUTOMATIC1111/TorchDeepDanbooru](https://github.com/AUTOMATIC1111/TorchDeepDanbooru)

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
On Mac Mini M4, MLX inference Speed:

```
1.7 seconds per image
```
