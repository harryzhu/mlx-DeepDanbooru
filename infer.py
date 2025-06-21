import time
import numpy as np
from PIL import Image, ImageDraw

import mlx.core as mx
from mlxDeepDanBooru.mlx_deep_danbooru_model import mlxDeepDanBooruModel


model_path = "models/model-resnet_custom_v3_mlx.npz"
tags_path = 'models/tags-resnet_custom_v3_mlx.npy'

mlx_dan = mlxDeepDanBooruModel()
mlx_dan.load_weights(model_path)
mx.eval(mlx_dan.parameters())


model_tags = np.load(tags_path)
print(f'total tags: {len(model_tags)}')

def danbooru_tags(fpath):
    tags = []
    pic = Image.open(fpath).convert("RGB").resize((512, 512))
    a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

    x = mx.array(a)
    y = mlx_dan(x)[0]

    for n in range(10):
        mlx_dan(x)
    for i, p in enumerate(y):
        if p >= 0.5:
            #print(model_tags[i].item(), p)
            tags.append(model_tags[i].item())

    return tags

image_count = 0
def image_infer(fpath):
    global image_count
    tags = danbooru_tags(fpath)
    image_count += 1
    return tags


t1 = time.time()
tags_1 = image_infer("example/1.png")
tags_2 = image_infer("example/2.png")

t2 = time.time()

print(tags_1)
print(tags_2)
# print(tags_3)
# print(tags_4)
# print(tags_5)

print("-----------")
print(f'infer speed(with mlx): {(t2 - t1)/image_count} seconds per image')






