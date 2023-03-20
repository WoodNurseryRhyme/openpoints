import os
import glob
import h5py
import numpy as np
import pickle
import logging
import json
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS
import open3d
import random

@DATASETS.register_module()
class PCNCls(Dataset):
    classes = ['airplane',
               'cabinet',
               'car',
               'chair',
               'lamp',
               'sofa',
               'table',
               'watercraft'
               ]

    def __init__(self,
                 num_points=2048,
                 data_dir="/DATA/mty/topology_aware_completion/data/PCN",
                 split='train',
                 transform=None
                 ):
        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.partition = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.category_file = '/DATA/mty/topology_aware_completion/data/PCN/PCN.json'

        self.partial_path = '/DATA/mty/topology_aware_completion/data/PCN/%s/partial/%s/%s/%02d.pcd'
        self.n_renderings = 8 if self.partition == 'train' else 1
        
        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())

        self.data_file, self.label = self._load_data(data_dir, self.partition, self.dataset_categories)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.partition} data')
        self.transform = transform

    def _load_data(self, data_dir, partition, dataset_categories):
        all_data_file = []
        all_label = []

        idx = 0
        
        for index, dc in enumerate(dataset_categories):
                samples = dc[partition]
                for s in samples:
                    """
                    for dense gt
                    /DATA/mty/topology_aware_completion/data/PCN/%s/complete/%s/%s.pcd
                    """
                    name = os.path.join(data_dir, partition, 'complete', dc['taxonomy_id'], '%s.pcd' % s)

                    """
                    for partial 
                    /DATA/mty/topology_aware_completion/data/PCN/%s/partial/%s/%s/%02d.pcd
                    """
                    # name = [ self.partial_path % (partition, dc['taxonomy_id'], s, i)
                    #         for i in range(self.n_renderings)
                    #     ]

                    """
                    for completion output
                    """
                    # name = os.path.join(data_dir, dc['taxonomy_id'], 'Model%02d-Dense.pcd' % idx)
                    # idx += 1

                    all_data_file.append(name)
                    all_label.append(np.int64(index))
        # from ipdb import set_trace
        # set_trace()
        # all_data_file = np.concatenate(all_data_file, axis=0)
        # all_label = np.concatenate(all_label, axis=0).squeeze(-1)

        return all_data_file, all_label
   
    def __getitem__(self, item):
        rand_idx = random.randint(0, self.n_renderings - 1) if self.partition=='train' else 0
        file_path = self.data_file[item]
        if type(file_path) == list:
            file_path = file_path[rand_idx]

        pc = open3d.io.read_point_cloud(file_path)
        pointcloud = np.array(pc.points).astype('float32')[:self.num_points]

        # pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    def __len__(self):
        return len(self.data_file)

    @property
    def num_classes(self):
        return np.max(self.label) + 1

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """