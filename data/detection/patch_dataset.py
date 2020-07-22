import os
import math
import pydicom
import itertools
from enum import Enum
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils import compute_iou_matrix
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


class PatchBasedDataset(Dataset):
    """
    Dataset class that builds internal utility for patch-based learning.
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
        - training: bool, training mode (if true, apply data augmentation).
        - filter: (str, PBFilter), filter mode. Value can be PBFilter.NONE
        for no filtering, PBFilter.POSITIVE_IMAGE for filtering only
        defective images and PBFilter.POSITIVE_PATCH for filtering only
        patches containing defects. If provided as a string, must be none,
        positive_image or positive_patch respectively.
        - keep_gt_threshold: float, threshold under which a part of a
        defect contained in a given patch is ignored.
        - seed: int, a given seed. This seed is used for data augmentation
        (flip, rotation).
    """

    def __init__(
        self,
        path,
        subset,
        patch_size=256,
        mode='grayscale',
        bits=8,
        mean=[0.5, 0.5, 0.5],
        std=[1.0, 1.0, 1.0],
        training=False,
        filter=PBFilter.NONE,
        keep_gt_threshold=0.2,
        seed=None
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
            - training: bool, training mode (if true, apply data augmentation).
            - filter: (str, PBFilter), filter mode. Value can be PBFilter.NONE
            for no filtering, PBFilter.POSITIVE_IMAGE for filtering only
            defective images and PBFilter.POSITIVE_PATCH for filtering only
            patches containing defects. If provided as a string, must be none,
            positive_image or positive_patch respectively.
            - keep_gt_threshold: float, threshold under which a part of a
            defect contained in a given patch is ignored.
            - seed: int, a given seed. This seed is used for data augmentation
            (flip, rotation).
        """
        self.path = path
        self.data_images = os.path.join(path, 'images')
        self.data_labels = os.path.join(path, 'labels')

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
        self.training = training
        self.patch_size = patch_size

        self.filter = get_PBFilter(filter)
        if self.filter == PBFilter.POSITIVE_IMAGE:
            self.filter_with_positive_image = True
            self.filter_with_positive_patch = False
        elif self.filter == PBFilter.POSITIVE_PATCH:
            self.filter_with_positive_image = True
            self.filter_with_positive_patch = True
        else:
            self.filter_with_positive_image = False
            self.filter_with_positive_patch = False

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
                raise ValueError("mean expected to be single value"
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
                raise ValueError("std expected to be single value"
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

        labels = []
        for image in self.image_files:
            label = image.replace('.jpg', '.txt').replace(
                '.jpeg', '.txt').replace('.png', '.txt').replace(
                    '.dcm', '.txt')
            labels.append(label)
        self.labels = labels

        gt_boxes = []
        for k, filename in enumerate(labels):
            with open(os.path.join(self.data_labels, filename)) as f:
                for line in f.readlines():
                    words = line.split(' ')
                    assert(len(words) == 5)
                    # category = words[0]
                    xmin = int(words[1])
                    xmax = int(words[2])
                    ymin = int(words[3])
                    ymax = int(words[4])
                    gt_boxes.append(
                        [xmin, ymin, xmax, ymax,
                         (xmax-xmin+1)*(ymax-ymin+1), k])

        self.gt_boxes = gt_boxes

        # filter positive images: images that contains defects
        if self.filter_with_positive_image:
            tmp = np.array(self.gt_boxes)
            valid_idx = np.unique(tmp[:, 5])
            # filter images
            self.image_files = [a for a, b in zip(
                self.image_files,
                np.arange(0, len(self.image_files), 1)
                ) if b in valid_idx]
            # filter labels
            self.labels = [a for a, b in zip(
                self.labels,
                np.arange(0, len(self.image_files), 1)
                ) if b in valid_idx]
            c, k = 0, valid_idx[0]
            # reset the corresponding image id to all boxes
            for box in self.gt_boxes:
                if box[5] > k:
                    k = box[5]
                    c += 1
                box[5] = c

        self.image_sizes = _get_image_dimensions(
            self.data_images, self.image_files,
            self.mode, self.bits_per_channel)
        self.patches, self.patch_grids = _get_patches_coords(
            self.image_sizes, patch_size)

        if self.filter_with_positive_patch:
            self.positive_patches, self.gt_box_positive, self.gt_boxes\
                 = _filter_positive_samples(
                    self.patches, self.gt_boxes,
                    len(self.image_files), iou_threshold=keep_gt_threshold)
            self.patches = self.positive_patches
        else:
            self.positive_patches, self.gt_box_positive, self.gt_boxes\
                = _associate_bbox_to_patch(
                    self.patches, self.gt_boxes,
                    len(self.image_files), iou_threshold=keep_gt_threshold)
            self.patches = self.positive_patches

        self.randgen = np.random.RandomState(seed=seed)

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
            - torch.LongTensor, mask tensor of size H x W
            - list[int], patch meta_data: x1 y1 x2 y2 area image_id
        """
        # print(len(self.gt_box_positive))
        patch = self.patches[index]
        image = _load_image(self.data_images, self.image_files[patch[5]],
                            bits=self.bits_per_channel, mode=self.mode)

        do_flip = self.randgen.rand(2, 1)
        rotate_angle = self.randgen.choice((0, 90, 180, 270))
        trsf_crop = CropTransform(
            patch[0], patch[1], self.patch_size, self.patch_size)
        trsf_hflip = transforms.RandomHorizontalFlip(p=1.0)
        trsf_vflip = transforms.RandomVerticalFlip(p=1.0)
        trsf_rotate = RotationTransform(rotate_angle)
        trsf_tensor = transforms.ToTensor() \
            if self.bits_per_channel == 8 else ToTensor()
        trsf_normalize = transforms.Normalize(self.mean, self.std)

        trsf_compose = [trsf_crop]
        if self.training is True:
            trsf_compose.append(trsf_rotate)
            if do_flip[0, 0] >= 0.5:
                trsf_compose.append(trsf_hflip)
            if do_flip[1, 0] >= 0.5:
                trsf_compose.append(trsf_vflip)
        trsf_compose.append(trsf_tensor)
        trsf_compose.append(trsf_normalize)

        data_transforms = transforms.Compose(trsf_compose)
        image = data_transforms(image)

        bboxes = self.gt_box_positive[index]

        pad_v = max(patch[3] - self.image_sizes[patch[5]][1], 0)
        pad_h = max(patch[2] - self.image_sizes[patch[5]][0], 0)

        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        if pad_v > 0:
            mask[self.patch_size-int(pad_v)-1:self.patch_size, :] = 255
        if pad_h > 0:
            mask[:, self.patch_size-int(pad_h)-1:self.patch_size] = 255

        for b in bboxes:
            x1, y1, x2, y2, _, _ = self.gt_boxes[b]
            x1 = max(x1-patch[0], 0)
            y1 = max(y1-patch[1], 0)
            x2 = min(x2-patch[0], self.patch_size)
            y2 = min(y2-patch[1], self.patch_size)
            mask[y1:y2, x1:x2] = 1  # 127  to test/visualize
        mask = Image.fromarray(mask, mode='L')

        if self.training is True:
            mask = trsf_rotate(mask)
            if do_flip[0, 0] >= 0.5:
                mask = trsf_hflip(mask)
            if do_flip[1, 0] >= 0.5:
                mask = trsf_vflip(mask)
        mask = torch.LongTensor(np.array(mask).astype(np.uint8))

        # print("Image {} with patch grid of {} by {}".format(
        #     self.image_files[patch[5]],
        #     self.patch_grids[patch[5]][0],
        #     self.patch_grids[patch[5]][1]))
        # patch = torch.LongTensor(np.array(patch).astype(np.int32))
        return image, mask, patch

    def get_raw_item(self, index):
        """
        A raw version of __getitem__ that returns PIL.Image instead of
        normalized tensors. Useful to visualize image along with mask.

        Params:
            - index: int, a given index.

        Returns:
            - PIL.Image, image of size H x W
            - PIL.Image, mask of size H x W
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

        bboxes = self.gt_box_positive[index]

        pad_v = max(patch[3] - self.image_sizes[patch[5]][1], 0)
        pad_h = max(patch[2] - self.image_sizes[patch[5]][0], 0)

        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        if pad_v > 0:
            mask[self.patch_size-int(pad_v)-1:self.patch_size, :] = 255
        if pad_h > 0:
            mask[:, self.patch_size-int(pad_h)-1:self.patch_size] = 255

        for b in bboxes:
            x1, y1, x2, y2, _, _ = self.gt_boxes[b]
            x1 = max(x1-patch[0], 0)
            y1 = max(y1-patch[1], 0)
            x2 = min(x2-patch[0], self.patch_size)
            y2 = min(y2-patch[1], self.patch_size)
            mask[y1:y2, x1:x2] = 1  # 127  to test/visualize
        mask = Image.fromarray(mask, mode='L')

        return image, mask, patch

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        gt = [item[1] for item in batch]
        patches = [item[2] for item in batch]
        images = torch.stack(images, dim=0)
        gt = torch.stack(gt, dim=0)
        # patches = torch.stack(patches, dim=0)

        return [images, gt, patches]

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


def _associate_bbox_to_patch(patch_coords, gt_boxes,
                             n_images, iou_threshold=0.5):
    """
    Locate gt boxes inside patches and associate them.

    Params:
        - patch_coords: list, list of patch coordinates for all images
        - gt_boxes: list, list of gt boxes
        - n_images: int, number of images in the dataset
        - iou_threshold: float, threshold under which part of a defect
        belonging to a given patch is ignore for this patch.

    Returns:
        - list, list of patch coordinates
        - list, list of gt boxes associated to patch coordinates
        - list, list of gt boxes (the input gt boxes but sorted by image)
    """
    patches_per_image = [[_ for _ in patch_coords if _[5] == k]
                         for k in range(n_images)]
    gt_bbox_per_image = [[_ for _ in gt_boxes if _[5] == k]
                         for k in range(n_images)]

    gt_box_to_patch, idx = [], 0
    for patches, gt_bbox in zip(patches_per_image, gt_bbox_per_image):
        if len(gt_bbox) > 0:
            _, _, inter_over_area2 = compute_iou_matrix(patches, gt_bbox)
            bb = [np.nonzero(inter_over_area2[p] > iou_threshold)
                  for p in np.arange(len(patches))]
            bb = [(np.array(elements)+idx).tolist()[0] for elements in bb]
            gt_box_to_patch.append(bb)
            idx += len(gt_bbox)
        else:
            gt_box_to_patch.append([[]]*len(patches))

    all_patches = list(itertools.chain.from_iterable(patches_per_image))
    gt_box_per_patch = list(itertools.chain.from_iterable(gt_box_to_patch))

    return all_patches, gt_box_per_patch,\
        list(itertools.chain.from_iterable(gt_bbox_per_image))


def _filter_positive_samples(patch_coords, gt_boxes,
                             n_images, iou_threshold=0.5):
    """
    Locate gt boxes inside patches and associate them. Also filter out patches
    with no gt boxes inside.

    Params:
        - patch_coords: list, list of patch coordinates for all images
        - gt_boxes: list, list of gt boxes
        - n_images: int, number of images in the dataset
        - iou_threshold: float, threshold under which part of a defect
        belonging to a given patch is ignore for this patch.

    Returns:
        - list, list of patch coordinates that contains defects
        - list, list of gt boxes associated to those patch coordinates
        - list, list of gt boxes (the input gt boxes but sorted by image)
    """
    patches_per_image = [[_ for _ in patch_coords if _[5] == k]
                         for k in range(n_images)]
    gt_bbox_per_image = [[_ for _ in gt_boxes if _[5] == k]
                         for k in range(n_images)]

    positive_patches = []
    negative_patches = []
    gt_box_to_positive, idx = [], 0
    for patches, gt_bbox in zip(patches_per_image, gt_bbox_per_image):
        if len(gt_bbox) > 0:
            _, _, inter_over_area2 = compute_iou_matrix(patches, gt_bbox)
            pos = np.sum(inter_over_area2 > iou_threshold, axis=1) > 0
            positive_patches.append([a for a, b in zip(patches, pos) if b])
            negative_patches.append([a for a, b in zip(patches, pos) if not b])
            bb = [np.nonzero(inter_over_area2[p] > iou_threshold)
                  for p in np.arange(len(patches))[pos]]
            bb = [(np.array(elements)+idx).tolist()[0] for elements in bb]
            gt_box_to_positive.append(bb)
            idx += len(gt_bbox)
        else:
            negative_patches.append([a for a in patches])

    positive_patches = list(itertools.chain.from_iterable(positive_patches))
    negative_patches = list(itertools.chain.from_iterable(negative_patches))
    gt_box_to_positive = list(
        itertools.chain.from_iterable(gt_box_to_positive))

    return positive_patches, gt_box_to_positive,\
        list(itertools.chain.from_iterable(gt_bbox_per_image))


def display_patches(ds: PatchBasedDataset, rows=2, cols=2, offset=0):
    le = len(ds)
    if le == 0:
        return
    if le < rows * cols:
        val = int(math.floor(math.sqrt(le)))
        rows, cols = val, val

    fig, axes = plt.subplots(
        rows, 2*cols, figsize=(cols*2, rows), constrained_layout=True)
    fig.suptitle('Tile map of crops', fontsize=14)

    ax = axes.ravel()
    for i in range(0, rows):
        for j in range(0, cols):
            k = i*2*cols+2*j
            image, mask, _ = ds.get_raw_item(i*cols+j+offset)
            ax[k].imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
            ax[k+1].imshow(mask, cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[k].axis('off')
            ax[k+1].axis('off')
    plt.show()
