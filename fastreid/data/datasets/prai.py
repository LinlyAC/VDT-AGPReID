# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
import pdb
from glob import glob

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

import pickle
import pdb

__all__ = ['PRAI', ]


@DATASET_REGISTRY.register()
class PRAI(ImageDataset):
    """PRAI
    """
    dataset_dir = "PRAI-1581"
    dataset_name = 'prai'

    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.data_dir = '/home/viu_user/Documents/QuanZhang/datasets/PRAI-1581/'
        self.img_path = os.path.join(self.data_dir, 'images')
        self.label_path = os.path.join(self.data_dir, 'partitions.pkl')
        with open(self.label_path, 'rb') as f:
            label = pickle.load(f)
        self.train_label = label['trainval_im_names']
        test_label = label['test_im_names']
        self.query_label, self.gallery_label = [], []
        for i in range(len(label['test_marks'])):
            if label['test_marks'][i] == 0:
                self.query_label.append(test_label[i])
            else:
                self.gallery_label.append(test_label[i])

        # required_files = [self.train_path]
        # self.check_before_run(required_files)
        self.cam_index = 1
        train = self.process_label(self.train_label, is_train=True)
        query = self.process_label(self.query_label, is_train=False)
        gallery = self.process_label(self.gallery_label, is_train=False)

        super().__init__(train, query, gallery, **kwargs)

    def process_label(self, img_paths, is_train=True):
        data = []
        for img_path in img_paths:
            pid = int(img_path.split('_')[0])
            # camid = self.cam_index
            camid = int(img_path.split('_')[1])
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((os.path.join(self.img_path, img_path), pid, camid))
            self.cam_index = self.cam_index + 1

        return data
