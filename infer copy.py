# load libraries
import glob
import torch
from PIL import Image, ImageDraw
import numpy as np
from typing import Any, Dict, Union

import mlx.core as mx
from mlxDeepDanbooru import mlxdan

from TorchDeepDanbooru import deep_danbooru_model

from typing import Union
import os
import time
import json
from mlx.utils import tree_unflatten

def save_npy(fpath, npy):
    with open(f'm/mlx_{fpath}', "w") as fw:
        fw.write(str(npy))

pth_dan = deep_danbooru_model.DeepDanbooruModel()
checkpoint = torch.load('_models/deepdanbooru/model-resnet_custom_v3.pt',map_location="cpu",weights_only=True)
pth_dan.load_state_dict(checkpoint)

pth_tags = pth_dan.tags


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))

mlx_dan = mlxdan.MLXDeepDanbooruModel()
mx.eval(mlx_dan.parameters())
#print(mlx_dan)


print("[INFO] Loading")
torch_weights = torch.load('_models/deepdanbooru/model-resnet_custom_v3.pt',map_location="cpu")

for k, v in torch_weights.items():
    print(f'pth: {k}')
    if k != "tags":
        #print(f'pth: {k}')
        if k.find(".weight") > 0:
            #print(f'pth: {k} ===> {v}')
            #print(f'pth: {k} ===> {v.shape}')
            #t_359 = t_358.transpose(*[0, 3, 1, 2]) 
            v2 = torch_to_mx(v, dtype='float32')
            v3 = v2.transpose(*[0, 2, 3, 1])
            #print(f'mlx: {k} ===> {v3}')
            #print(f'mlx: {k} ===> {v3.shape}')
            mlx_dan[k] = v3
        #print(f'pth: {k}')
        if k.find(".bias") > 0:
            #print(f'pth: {k} ===> {v}')
            #t_359 = t_358.transpose(*[0, 3, 1, 2]) 
            v2 = torch_to_mx(v, dtype='float32')
            #print(v3.shape)
            #print(f'mlx: {k} ===> {v2}')
            mlx_dan[k] = v2
        #break

# print(type(pth_tags))
#mlx_dan.load_tags(pth_tags)
# print(pth_tags[0:10])

print("[INFO] Converting")

print("[INFO] Saving")
mlx_path = "m2222.npz"

#mlx_dan.save_weights(mlx_path)


print("\n\n\n\n[LOOOOOOAD] ============")
mlx_dan2 = mlxdan.MLXDeepDanbooruModel()


mlx_dan2.load_weights(mlx_path)

# f3items = list(weights.items())[0:3]
# for f3i in f3items:
#     print(f3i)
# print("[weightsweightsweightsweightsweights] ============")
# weights = tree_unflatten(list(weights.items()))

# mlx_dan.update(weights)
# mx.eval(mlx_dan.parameters())



def danbooru_tags(fpath):
    tags = []
    
    pic = Image.open(fpath).convert("RGB").resize((512, 512))

    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
    #print(f'a.shape::::: {a.shape}')

    x = mx.array(a)
    #print(f'x.shape::::: {x.shape}')

    y = mlx_dan2(x)[0]
    #print(f'{y}')
    #print(f'y.shape::::: {y.shape}')
    for n in range(10):
        mlx_dan2(x)
    for i, p in enumerate(y):
        if p >= 0.5:
            #print(pth_tags[i], p)
            tags.append(pth_dan.tags[i])

    return tags

t1 = time.time()
tags = danbooru_tags("2.png")
t2 = time.time()
print(f'mlx duration: {t2 - t1}')
print("--------")
print(tags)




