import os
import numpy as np

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
from util.crop import center_crop_arr_small, random_crop_arr
small_transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, 128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample1 = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample1)
        if self.target_transform is not None:
            target = self.target_transform(target)
        small_sample = small_transform_train(sample1)
        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, small_sample, filename #target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target
    
class SmallCachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
            small_moments = data['small_moments']
        else:
            moments = data['moments_flip']
            small_moments = data['small_moments_flip']

        return moments, small_moments, target


    
def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    files_caption_token = glob.glob(os.path.join(root, '*_*_token.npy'))
    num_data = len(files) - len(files_caption)
    print(num_data)
    add_num_data = len(files_caption_token)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        if ('token' in f) or ('one' in f) or ('orig' in f) or ('open' in f) or ('d' in f) or ('lfq' in f):
            continue
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        if int(k1) > num_data:
            print(f)
        n_captions[int(k1)] += 1
    return num_data, n_captions, files_caption_token

def get_feature_dir_len(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    
    num_data = len(files) - len(files_caption)
    print(num_data)
    
    return num_data
