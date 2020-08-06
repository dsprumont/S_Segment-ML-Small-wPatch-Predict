import os
import math
import pydicom
from enum import Enum
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .dataset import ToTensor


class RotationTransform:
    """
    Take a PIL image an return a rotated version as PIL Image.
    """
    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def __call__(self, image):
        return TF.rotate(image, self.rotation_angle)


class CropTransform:
    """
    Take a PIL image and return a crop region as PIL image.
    """
    def __init__(self, top, left, width, height):
        self.width = height  # width
        self.height = width  # height
        self.top = left  # top
        self.left = top  # left

    def __call__(self, image):
        return TF.crop(image, self.top, self.left, self.height, self.width)


class PBFilter(Enum):
    NONE = 0
    POSITIVE_IMAGE = 1
    POSITIVE_PATCH = 2


def get_PBFilter(name):
    _str2enum = {
        'none': PBFilter.NONE,
        'positive_image': PBFilter.POSITIVE_IMAGE,
        'positive_patch': PBFilter.POSITIVE_PATCH
    }
    if isinstance(name, (str,)):
        return _str2enum.get(name, PBFilter.NONE)
    elif isinstance(name, (PBFilter,)):
        return name
    else:
        raise TypeError("Expected str or PBFilter"
                        ", got {} instead".format(type(name)))


class InferencePatchBasedDataset(Dataset):
    """
    Dataset class that builds internal utility for patch-based inference.
    Images are loaded and divided into patches of given size.

    Params:
        - path: str, a path to the dataset repository
        - subset: str, a subset name (train, val or test) that we want to
        sample from.
        - patch_size: int, size of the produced patches.
        - mode: str, the image mode (grayscale or rgb).
        - bits: int, the bit-encoding of image (png/jpeg: 8, dicom: 16).
        - mean: (float, list[float]), the mean of dataset pixels
        (this mean applies to Pytorch.Tensor conversion of image).
        - std: (float, list[float]), the variance of dataset pixels
        (this mean applies to Pytorch.Tensor conversion of image).
    """

    def __init__(
        self,
        path,
        subset,
        patch_size=256,
        mode='grayscale',
        bits=8,
        mean=[0.5, 0.5, 0.5],
        std=[1.0, 1.0, 1.0]
    ):
        """
        Params:
            - path: str, a path to the dataset repository
            - subset: str, a subset name (train, val or test) that we want to
            sample from.
            - patch_size: int, size of the produced patches.
            - mode: str, the image mode (grayscale or rgb).
            - bits: int, the bit-encoding of image (png/jpeg: 8, dicom: 16).
            - mean: (float, list[float]), the mean of dataset pixels
            (this mean applies to Pytorch.Tensor conversion of image).
            - std: (float, list[float]), the variance of dataset pixels
            (this mean applies to Pytorch.Tensor conversion of image).
        """
        self.path = path
        self.data_images = os.path.join(path, 'images')

        if isinstance(bits, (int,)) and bits in (8, 16):
            self.bits_per_channel = bits
        else:
            raise ValueError("bits expected an integer value of 8 or 16,\
                 got {} instead".format(bits))

        if isinstance(mode, (str,)) and mode in ['grayscale', 'rgb']:
            if self.bits_per_channel == 16 and mode == 'rgb':
                raise ValueError("rgb mode is not supported for 16bits images")
            else:
                self.mode = {'grayscale': 'L', 'rgb': 'RGB'}[mode]
        else:
            raise ValueError("mode expected to be in [grayscale, rgb],\
                 got {} instead".format(mode))

        self.in_channels = 1 if mode == 'L' else 3
        self.patch_size = patch_size

        # either,
        # 8bits grayscale
        # 8bits rgb
        # 16bits grayscale
        if isinstance(mean, (int, float)):
            if self.mode == 'RGB':
                self.mean = (mean, mean, mean)
            else:
                self.mean = (mean,)
        elif isinstance(mean, (list,)) \
                and all(isinstance(m, (int, float,)) for m in mean):
            if self.mode == 'RGB':
                self.mean = mean
            else:
                self.mean = (mean[0],)
                print("Warning: mean expected to be single value"
                      " with grayscale mode, got tuple instead")
        else:
            raise ValueError("mean expected to be int, float or "
                             "list[int, float], got {} instead".format(mean))

        if isinstance(std, (int, float,)):
            if self.mode == 'RGB':
                self.std = (std, std, std)
            else:
                self.std = (std,)
        elif isinstance(std, (list,)) \
                and all(isinstance(m, (int, float)) for m in std):
            if self.mode == 'RGB':
                self.std = std
            else:
                self.std = (std[0],)
                print("Warning: std expected to be single value"
                      " with grayscale mode, got tuple instead")
        else:
            raise ValueError("std expected to be int, float or "
                             "list[int, float], got {} instead".format(std))

        subset_file = os.path.join(path, '{}.txt'.format(subset))
        if not os.path.exists(subset_file):
            raise FileExistsError("the file <{}> does not "
                                  "exist".format(subset_file))
        with open(subset_file) as f:
            images = [item.split('\n')[0] for item in f.readlines()]
        self.image_files = images

        self.image_sizes = _get_image_dimensions(
            self.data_images, self.image_files,
            self.mode, self.bits_per_channel)
        self.patches, self.patch_grids = _get_patches_coords(
            self.image_sizes, patch_size)

    def __len__(self):
        return len(self.patches)

    def get_name(self, index):
        """
        Return the filename of image at given index.
        """
        return self.image_files[index]

    def get_size(self, index):
        """
        Return the dimensions of image at given index.
        """
        return self.image_sizes[index]

    def __getitem__(self, index):
        """
        Params:
            - index: int, a given index.

        Returns:
            - torch.FloatTensor, image tensor of size H x W
            - list[int], patch meta_data: x1 y1 x2 y2 area image_id
        """
        # print(len(self.gt_box_positive))
        patch = self.patches[index]
        image = _load_image(self.data_images, self.image_files[patch[5]],
                            bits=self.bits_per_channel, mode=self.mode)

        trsf_crop = CropTransform(
            patch[0], patch[1], self.patch_size, self.patch_size)
        trsf_tensor = transforms.ToTensor() \
            if self.bits_per_channel == 8 else ToTensor()
        trsf_normalize = transforms.Normalize(self.mean, self.std)

        trsf_compose = [trsf_crop]
        trsf_compose.append(trsf_tensor)
        trsf_compose.append(trsf_normalize)

        data_transforms = transforms.Compose(trsf_compose)
        image = data_transforms(image)

        return image, patch

    def get_raw_item(self, index):
        """
        A raw version of __getitem__ that returns PIL.Image instead of
        normalized tensors. Useful to visualize image along with mask.

        Params:
            - index: int, a given index.

        Returns:
            - PIL.Image, image of size H x W
            - list[int], patch meta_data: x1 y1 x2 y2 area image_id
        """
        # print(len(self.gt_box_positive))
        patch = self.patches[index]
        image = _load_image(self.data_images, self.image_files[patch[5]],
                            bits=self.bits_per_channel, mode=self.mode)

        trsf_crop = CropTransform(
            patch[0], patch[1], self.patch_size, self.patch_size)
        image = trsf_crop(image)
        if self.bits_per_channel == 16:
            data = list(image.getdata())
            width, height = image.size
            np_array = np.array(data, dtype=np.float32).reshape(height, width)
            image = Image.fromarray(np_array / 255)

        return image, patch

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        patches = [item[1] for item in batch]
        images = torch.stack(images, dim=0)

        return [images, patches]

    def __str__(self):
        return "PatchBasedDataset\npatch_size={}\nImage_count={}\n"\
            "Patches_count={}".format(
                self.patch_size, len(self.image_files), len(self.patches))


def _get_patches_coords(image_sizes, patch_size):
    """
    Take a list of image sizes and defined the number of columns, rows, as
    well as the coordinates of top-left and bottom-right corner of each
    patch that belongs to each image in the list.
    """
    patches = []
    patch_grids = []
    stride = int(patch_size / 2)
    for k, dims in enumerate(image_sizes):
        width, height = dims

        n_w = math.ceil(width / patch_size) * 2 - 1 \
            - int((width % patch_size) < stride)
        n_h = math.ceil(height / patch_size) * 2 - 1 \
            - int((height % patch_size) < stride)
        n_w = n_w if n_w > 0 else 1
        n_h = n_h if n_h > 0 else 1

        for j in range(n_h):
            for i in range(n_w):
                patches.append([i*stride, j*stride,
                               i*stride+patch_size-1, j*stride+patch_size-1,
                               patch_size*patch_size, k])
        patch_grids.append([n_w, n_h])
    return patches, patch_grids


def _load_image(path, filename, bits, mode):
    """
    Load an image from the given path+filename based on bits and mode.
    """
    if filename.rsplit('.')[1].lower() == 'dcm':
        ds = pydicom.dcmread(os.path.join(path, filename))
        m = ('I;16' if bits == 16 else 'L') if mode == 'L' else 'RGB'
        image = Image.frombuffer(
            m, (ds.Columns, ds.Rows), ds.PixelData, 'raw', m, 0, 1)
    else:
        image = Image.open(os.path.join(path, filename)).convert(mode)
    return image


def _get_image_dimensions(path, image_files, mode='L', bits=8):
    dimensions = []
    for filename in image_files:
        image = _load_image(path, filename, bits, mode)
        width, height = image.size
        dimensions.append([width, height])
    return dimensions
