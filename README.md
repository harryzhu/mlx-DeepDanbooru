# mlxDeepDanbooru
Pure MLX implementation of DeepDanbooru for Apple Silicon Chips: M1, M2, M3, M4

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
 Go to `https://huggingface.co/hazhu/mlxDeepDanbooru` then explore `models` floder,
 and download `model-resnet_custom_v3_mlx.npz` and `tags-resnet_custom_v3_mlx.npy`
 and put them into models folder.

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
