import os
import time
import glob
import numpy as np
from PIL import Image, ImageDraw

import mlx.core as mx
from mlxDeepDanBooru.mlx_deep_danbooru_model import mlxDeepDanBooruModel

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from copy import deepcopy

ROOTDIR = os.path.dirname(os.path.abspath(__file__))
IMAGEDIR = f'{ROOTDIR}/example'


model_path = f"{ROOTDIR}/models/model-resnet_custom_v3_mlx.npz"
tags_path = f'{ROOTDIR}/models/tags-resnet_custom_v3_mlx.npy'

mlx_dan = mlxDeepDanBooruModel()
mlx_dan.load_weights(model_path)
mx.eval(mlx_dan.parameters())


model_tags = np.load(tags_path)
#print(f'total tags: {len(model_tags)}')

def danbooru_tags(fpath):
    results = {}
    tags = []

    pic = Image.open(fpath).convert("RGB").resize((512, 512))
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

    x = mx.array(a)
    y = mlx_dan(x)[0]

    try:
        for n in range(10):
            mlx_dan(x)
        for i, p in enumerate(y):
           if p >= 0.55:             
                #print(model_tags[i].item(), p)
                tags.append(model_tags[i].item())
    except Exception as err:
        print(err)

    results[fpath] = tags
    return results


def image_infer(fpath):
    tags = danbooru_tags(fpath)
    return tags

t1 = time.time()

tags_1 = image_infer(f'{IMAGEDIR}/1.png')
tags_2 = image_infer(f'{IMAGEDIR}/2.png')

t2 = time.time()

print(tags_1)
print(tags_2)

print(f'2 images: infer speed(with mlx): {(t2 - t1)/2} seconds per image')






    



