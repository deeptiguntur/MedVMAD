import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'brain':
        obj_list = ['brain']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='train'):
        print(mode)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
        img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # if anomaly == 0:
        #     img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        # else:
        #     if os.path.isdir(os.path.join(self.root, mask_path)):
        #         # just for classification not report error
        #         img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        #     else:
        #         img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
        #         img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    