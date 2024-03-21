# encoding: utf-8

import os
import os.path as osp
import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

import pdb

__all__ = ['CARGO', ]


@DATASET_REGISTRY.register()
class CARGO(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = 'XXX'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid -= 1  # index starts from 0

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data


@DATASET_REGISTRY.register()
class CARGO_AA(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_aa'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = 'XXX'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid -= 1  # index starts from 0
            if viewid == 'Ground':
                continue

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data


@DATASET_REGISTRY.register()
class CARGO_GG(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_gg'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = 'XXX'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            if viewid == 'Aerial':
                continue
            camid -= 1  # index starts from 0

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data


@DATASET_REGISTRY.register()
class CARGO_AG(ImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo_ag'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = 'XXX'

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        train = self.process_dir(self.train_dir, is_train=True)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = []
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        data = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            camid = 1 if camid <= 5 else 2
            camid -= 1  # index starts from 0

            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid, viewid))
        return data
