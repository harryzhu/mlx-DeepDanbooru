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

worker_count = os.cpu_count()
# worker_count depends on your unified-memory size
# if oom, decrease the number

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


def batch_infer(image_list):
    workers = min(len(image_list), worker_count)
    print(f'workers: {workers}: {os.cpu_count()}')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        process_results = list(executor.map(image_infer, image_list))
        return process_results



if __name__ == '__main__':
    image_list = []
    for root, dirs, files in os.walk(IMAGEDIR, True):
        for file in files:
            if not file[-4:].lower() in [".png", ".jpg", "jpeg"]:
                continue
            fpath = os.path.join(root, file).replace("\\","/")
            image_list.append(fpath)
    
    #print(image_list)


    t1 = time.time()
    lines = batch_infer(image_list)
    t2 = time.time()

    for line in lines:
        print(line)
        print("-----------")

    print(f'{len(image_list)} images: infer speed(with mlx): {(t2 - t1)/len(image_list)} seconds per image')



