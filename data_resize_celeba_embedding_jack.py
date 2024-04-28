import argparse
import multiprocessing
import os
import shutil
from functools import partial
from io import BytesIO
from multiprocessing import Process, Queue
from os.path import exists, join
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import LSUNClass
from torchvision.transforms import functional as trans_fn
from tqdm import tqdm
import numpy as np

import h5py

h5_path = '/home/jack/ddpm/store/datasets/celeba_addendum/attributes_insight.hdf5'


def resize_and_convert(img, size, resample, quality=100):
    if size is not None:
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)

    buffer = BytesIO()
    img.save(buffer, format="webp", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img,
                    sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS,
                    quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(idx, img, sizes, resample):
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)
    return idx, out


class ConvertDataset(Dataset):
    def __init__(self, data, size) -> None:
        self.data = data
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, embedding = self.data[index]
        bytes = resize_and_convert(img, self.size, Image.LANCZOS, quality=100)
        return bytes, embedding


class ImageFolder(Dataset):
    def __init__(self, folder, ext='jpg'):
        super().__init__()
        paths = sorted([p for p in Path(f'{folder}').glob(f'*.{ext}')])
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.paths[index])
        img = Image.open(path)
        return img


class ImageAttributeFolder(Dataset):
    def __init__(self, folder, ext='jpg'):
        super().__init__()
        self.attr = h5py.File(h5_path)
        self.folder = folder
        self.file_names = sorted(list(self.attr.keys()))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        path = self.folder + '/' + file_name
        img = Image.open(path)
        embedding = self.attr[file_name]['embedding']
        return img, embedding


if __name__ == "__main__":
    from tqdm import tqdm

    out_path = 'datasets/celeba.lmdb'
    out_path = 'datasets/celeba_embeddings_v2.lmdb'
    in_path = 'datasets/celeba/celeba/img_align_celeba'
    ext = 'jpg'
    size = None

    #dataset = ImageFolder(in_path, ext)
    dataset = ImageAttributeFolder(in_path, ext)
    print('len:', len(dataset))
    dataset = ConvertDataset(dataset, size)
    loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=0,
                        collate_fn=lambda x: x,
                        shuffle=False)

    target = os.path.expanduser(out_path)
    if os.path.exists(target):
        shutil.rmtree(target)

    with lmdb.open(target, map_size=1024**4, readahead=False) as env:
        with tqdm(total=len(dataset)) as progress:
            b = 0
            for batch in loader:
                with env.begin(write=True) as txn:
                    j = 0
                    for p in batch:
                        i = b*len(batch)+j
                        img, embedding = p
                        key = f"{size}-{str(i).zfill(7)}".encode("utf-8")
                        # print(key)
                        txn.put(key, img)

                        embedding = np.array(embedding)
                        embedding_bytes = np.array(embedding).tobytes()
                        key = f"{size}-{str(i).zfill(7)}-embedding".encode("utf-8")
                        # print(key)
                        print(key, embedding.sum(), embedding[:4])
                        txn.put(key, embedding_bytes)
                        j += 1
                        progress.update()

                with env.begin(write=False) as txn:
                    j = 0
                    for p in batch:
                        i = b*len(batch)+j
                        key = f"{size}-{str(i).zfill(7)}-embedding".encode("utf-8")
                        embedding_bytes = txn.get(key)
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        print(key, embedding.sum(), embedding[:4])
                        j += 1
                import IPython ; IPython.embed()
                b += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(i).encode("utf-8"))
