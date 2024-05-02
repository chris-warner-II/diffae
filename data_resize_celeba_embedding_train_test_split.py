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

h5_path = 'store/datasets/celeba_addendum/attributes_insight.hdf5'


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
        img, embedding, landmark, file_name = self.data[index]
        bytes = resize_and_convert(img, self.size, Image.LANCZOS, quality=100)
        return bytes, embedding, landmark, file_name


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
        landmark = self.attr[file_name]['landmark_3d_68'] # CW

        return img, embedding, landmark, file_name


if __name__ == "__main__":
    from tqdm import tqdm

    #out_path = 'datasets/celeba.lmdb'
    out_path_test = 'store/datasets/diffae/datasets/celeba_embeddings_test.lmdb'
    out_path_train = 'store/datasets/diffae/datasets/celeba_embeddings_train.lmdb'
    in_path = 'store/datasets/celeba/celeba/img_align_celeba'
    ext = 'jpg'
    size = None


    # Pick out a few identities to go into the test set.
    # Don't include them in the training set.
    id_2_file = {}
    file_2_id = {}
    id_set = set()
    with open(f"{in_path}/../identity_CelebA.txt", "r") as file:
        for i,line in enumerate(file):
            parts = line.split()
            iden = int(parts[1])
            jpgfile = parts[0]
            #print(key,value)
            id_set.add(iden)
            file_2_id[jpgfile] = iden
            try:
                id_2_file[iden].append(jpgfile)
            except:
                id_2_file[iden] = []
                id_2_file[iden].append(jpgfile)

    id_test_list = list(np.arange(1,20)) # [] # see store/output/celeba-identity/face_identity_GT
    #
    file_test_list = []
    for id in id_test_list:
        file_test_list.extend( id_2_file[id] )




    check_bytes_match = False # print out embeddings and landmarks before converting
                              # to bytes and after conversion as sanity check

    #dataset = ImageFolder(in_path, ext)
    dataset = ImageAttributeFolder(in_path, ext)
    print('len:', len(dataset))
    dataset = ConvertDataset(dataset, size)
    loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=0,
                        collate_fn=lambda x: x,
                        shuffle=False)





    #import IPython; IPython.embed()

    # # Make the train.lmdb
    # target_train = os.path.expanduser(out_path_train)
    # if os.path.exists(target_train):
    #     shutil.rmtree(target_train)
    #
    # print('making train.lmdb')
    # with lmdb.open(target_train, map_size=1024**4, readahead=False) as env:
    #     with tqdm(total=len(dataset)) as progress:
    #         b = 0
    #         i1 = 0 # counter to build lmdb dataset
    #         i2 = 0 # counter to check that bytes match
    #         for batch in loader:
    #             with env.begin(write=True) as txn:
    #                 j = 0
    #                 for p in batch:
    #                     i = b*len(batch)+j
    #                     img, embedding, landmark, fname = p
    #                     if fname not in file_test_list:
    #                         key = f"{size}-{str(i1).zfill(7)}".encode("utf-8")
    #                         # print(key)
    #                         txn.put(key, img)
    #
    #                         key = f"{size}-{str(i1).zfill(7)}-fname".encode("utf-8")
    #                         txn.put(key, fname.encode("utf-8"))
    #                         if check_bytes_match: print(key, fname)
    #
    #                         embedding = np.array(embedding)
    #                         embedding_bytes = embedding.tobytes()
    #                         key = f"{size}-{str(i1).zfill(7)}-embedding".encode("utf-8")
    #                         if check_bytes_match: print(key, embedding.sum(), embedding[:4])
    #                         txn.put(key, embedding_bytes)
    #
    #                         landmark = np.array(landmark).ravel() # shape from (68,3) to (204,)
    #                         landmark_bytes = landmark.tobytes()
    #                         key = f"{size}-{str(i1).zfill(7)}-landmark".encode("utf-8")
    #                         if check_bytes_match: print(key, landmark.sum(), landmark.shape, landmark[:4])
    #                         txn.put(key, landmark_bytes)
    #                         i1 += 1
    #
    #                     j += 1
    #                     progress.update()
    #
    #             # Reading it back in from bytes to check that it matches expectations.
    #             if check_bytes_match:
    #                 with env.begin(write=False) as txn:
    #                     j = 0
    #                     for p in batch:
    #                         i = b*len(batch)+j
    #                         img, embedding, landmark, fname = p
    #                         if fname not in file_test_list:
    #                             #
    #                             key = f"{size}-{str(i2).zfill(7)}-fname".encode("utf-8")
    #                             fname_b = txn.get(key)
    #                             print(key, fname_b)
    #                             #
    #                             key = f"{size}-{str(i2).zfill(7)}-embedding".encode("utf-8")
    #                             embedding_bytes = txn.get(key)
    #                             embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    #                             print(key, embedding.sum(), embedding[:4])
    #                             #
    #                             key = f"{size}-{str(i2).zfill(7)}-landmark".encode("utf-8")
    #                             landmark_bytes = txn.get(key)
    #                             landmark = np.frombuffer(landmark_bytes, dtype=np.float32)
    #                             print(key, landmark.sum(), landmark[:4])
    #                             i2 += 1
    #                             #
    #                         j += 1
    #
    #                 assert i1 == i2
    #                 import IPython ; IPython.embed()
    #             b += 1
    #
    #     with env.begin(write=True) as txn:
    #         txn.put("length".encode("utf-8"), str(i1).encode("utf-8"))


    # Make the test.lmdb
    target_test = os.path.expanduser(out_path_test)
    if os.path.exists(target_test):
        shutil.rmtree(target_test)

    print('making test.lmdb')
    with lmdb.open(target_test, map_size=1024**4, readahead=False) as env:
        with tqdm(total=len(dataset)) as progress:
            b = 0
            i1 = 0
            i2 = 0
            for batch in loader:
                with env.begin(write=True) as txn:
                    j = 0
                    for p in batch:
                        i = b*len(batch)+j
                        img, embedding, landmark, fname = p
                        if fname in file_test_list:
                            key = f"{size}-{str(i1).zfill(7)}".encode("utf-8")
                            # print(key)
                            txn.put(key, img)

                            key = f"{size}-{str(i1).zfill(7)}-fname".encode("utf-8")
                            txn.put(key, fname.encode("utf-8"))
                            if check_bytes_match: print(key, fname)

                            embedding = np.array(embedding)
                            embedding_bytes = embedding.tobytes()
                            key = f"{size}-{str(i1).zfill(7)}-embedding".encode("utf-8")
                            if check_bytes_match: print(key, embedding.sum(), embedding[:4])
                            txn.put(key, embedding_bytes)

                            landmark = np.array(landmark).ravel() # shape from (68,3) to (204,)
                            landmark_bytes = landmark.tobytes()
                            key = f"{size}-{str(i1).zfill(7)}-landmark".encode("utf-8")
                            if check_bytes_match: print(key, landmark.sum(), landmark.shape, landmark[:4])
                            txn.put(key, landmark_bytes)
                            i1 += 1

                        j += 1
                        progress.update()

                # Reading it back in from bytes to check that it matches expectations.
                if check_bytes_match:
                    with env.begin(write=False) as txn:
                        j = 0
                        for p in batch:
                            i = b*len(batch)+j
                            img, embedding, landmark, fname = p
                            if fname in file_test_list:
                                #
                                key = f"{size}-{str(i2).zfill(7)}-fname".encode("utf-8")
                                fname_b = txn.get(key)
                                print(key, fname_b)
                                #
                                key = f"{size}-{str(i2).zfill(7)}-embedding".encode("utf-8")
                                embedding_bytes = txn.get(key)
                                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                                print(key, embedding.sum(), embedding[:4])
                                #
                                key = f"{size}-{str(i2).zfill(7)}-landmark".encode("utf-8")
                                landmark_bytes = txn.get(key)
                                landmark = np.frombuffer(landmark_bytes, dtype=np.float32)
                                print(key, landmark.sum(), landmark[:4])
                                i2 += 1
                                #
                            j += 1

                    assert i1 == i2
                    import IPython; IPython.embed()
                b += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(i1).encode("utf-8"))

