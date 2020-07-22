# common dependencies
import os
import random
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# deep learning dependencies
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms


class ToTensor():
    def __call__(self, image):
        if isinstance(image, (Image.Image,)):
            data = list(image.getdata())
            width, height = image.size
            np_array = np.array(data, dtype=np.float32).reshape(height, width)
            tensor = torch.from_numpy(
                np_array / (65535 if '16' in image.mode else 255))
            return tensor.unsqueeze(0) if len(image.getbands()) < 2 else tensor
        return image


class CustomDataset(Dataset):

    def __init__(
        self,
        data_path,
        subset,
        input_size=(224, 224),
        task='classification',
        mode='L',
        bits=8,
        training=False,
        scale=None,
        crop_size=None
    ):
        self.data_path = data_path
        self.data_images = os.path.join(data_path, subset)
        self.data_labels = os.path.join(data_path, 'labels')
        self.bits_per_channel = bits
        self.mode = mode if mode in ['L', 'RGB'] else 'RGB'
        self.in_channels = 1 if mode == 'L' else 3
        self.task = task \
            if task in ['classification', 'segmentation'] else 'classification'

        self.training = training
        self.scale = scale
        self.crop_size = crop_size \
            if crop_size is not None else input_size
        self.input_size = input_size

        images = os.listdir(self.data_images)
        images = [image for image in images
                  if image.rsplit('.')[1].lower()
                  in ['jpg', 'jpeg', 'png', 'dcm']]
        self.images = images
        labels = []
        for image in self.images:
            label = image.replace('.jpg', '.txt').replace(
                '.jpeg', '.txt').replace('.png', '.txt').replace(
                    '.dcm', '.txt')
            with open(os.path.join(self.data_labels, label)) as f:
                if self.task == 'classification':
                    labels.append(0 if len(f.readlines()) < 1 else 1)
                else:  # segmentation
                    labels.append(label)

        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print("LINK : {}".format(self.images[idx]))
        if self.images[idx].rsplit('.')[1].lower() == 'dcm':
            ds = pydicom.dcmread(os.path.join(
                self.data_images, self.images[idx]))
            m = ('I;16' if self.bits_per_channel == 16 else 'L') \
                if self.mode == 'L' else 'RGB'
            image = Image.frombuffer(
                m, (ds.Columns, ds.Rows), ds.PixelData, 'raw', m, 0, 1)
        else:
            image = Image.open(
                os.path.join(self.data_images, self.images[idx])
            ).convert(self.mode)
        label = self.labels[idx]

        if self.task == 'segmentation':
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
            # print("ORIGI: Image {} -- Mask {}".format(
            #     image.size, np.shape(mask)))
            with open(os.path.join(self.data_labels, label)) as f:
                for line in f.readlines():
                    words = line.split(' ')
                    assert(len(words) == 5)
                    # category = words[0]
                    xmin = int(words[1])
                    xmax = int(words[2])
                    ymin = int(words[3])
                    ymax = int(words[4])
                    mask[ymin:ymax, xmin:xmax] = 1  # category
            mask = Image.fromarray(mask, mode='L')

            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # ax = axes.ravel()
            # ax[0].imshow(image, cmap=plt.cm.gray)
            # ax[1].imshow(mask, cmap=plt.cm.gray)
            # ax[0].axis('off')
            # ax[1].axis('off')
            # fig.tight_layout()
            # plt.show()

        else:
            mask = None

        def _augmentation(
            image,
            mask,
            input_size,
            flip=False,
            scale=None,
            crop_size=None
        ):
            if input_size is not None:
                width, height = image.size  # original dimension
                w_max, h_max = input_size   # final dimension
                ar_from = width / height    # aspect ratio (original)
                ar_to = w_max / h_max       # aspect ratio (final)
                if ar_to > ar_from:
                    w_resize = int(np.floor(h_max * ar_from))
                    h_resize = h_max
                else:
                    w_resize = w_max
                    h_resize = int(np.floor(w_max / ar_from))
                image = image.resize(
                    (w_resize, h_resize),
                    Image.BICUBIC if self.bits_per_channel == 8
                    else Image.NEAREST)
                if mask is not None:
                    mask = mask.resize((w_resize, h_resize), Image.NEAREST)
                    # print("INPUT: Image {} -- Mask {}".format(
                    #     image.size, mask.size))

                    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    # ax = axes.ravel()
                    # ax[0].imshow(image, cmap=plt.cm.gray)
                    # ax[1].imshow(mask, cmap=plt.cm.gray)
                    # ax[0].axis('off')
                    # ax[1].axis('off')
                    # fig.tight_layout()
                    # plt.show()

            if flip:
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    if mask is not None:
                        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                        # print("FLIP : Image {} -- Mask {}".format(
                        #     image.size, mask.size))

            if scale is not None:
                smin, smax = scale
                coef = smin + random.random() * (smax - smin)
                new_size = (int(round(coef * width)),
                            int(round(coef * height)))
                image = image.resize(
                    new_size,
                    Image.BICUBIC if self.bits_per_channel == 8
                    else Image.NEAREST)
                if mask is not None:
                    mask = mask.resize(new_size, Image.NEAREST)
                    # print("SCALE: Image {} -- Mask {}".format(
                    #         image.size, mask.size))

            data_transforms = transforms.Compose([
                transforms.ToTensor() if self.bits_per_channel == 8
                else ToTensor(),
                transforms.Normalize(
                    [0.157] * self.in_channels if
                    self.in_channels > 1 else (0.157,),
                    [0.060] * self.in_channels if
                    self.in_channels > 1 else (0.060,))
            ])
            image = data_transforms(image)
            if mask is not None:
                mask = torch.LongTensor(np.array(mask).astype(np.uint8))
                # print("TORCH: Image {} -- Mask {}".format(
                #             image.shape, mask.shape))

                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # ax = axes.ravel()
                # ax[0].imshow(image.numpy().squeeze()*255, cmap=plt.cm.gray)
                # ax[1].imshow(mask.numpy()*255, cmap=plt.cm.gray)
                # ax[0].axis('off')
                # ax[1].axis('off')
                # fig.tight_layout()
                # plt.show()

            if crop_size is not None:
                height, width = image.shape[2], image.shape[1]
                vp = max(0, self.crop_size[0] - width)
                hp = max(0, self.crop_size[1] - height)
                # print(image.shape)
                image = nn.ZeroPad2d((0, hp, 0, vp))(image)
                # print(image.shape)
                h, w = image.shape[2], image.shape[1]
                off_w = random.randint(0, w - self.crop_size[0])
                off_h = random.randint(0, h - self.crop_size[1])
                image = image[
                        :,
                        off_w:off_w+self.crop_size[0],
                        off_h:off_h+self.crop_size[1]]
                if mask is not None:
                    # print(mask.shape)
                    mask = nn.ConstantPad2d((0, hp, 0, vp), 255)(mask)
                    # print(mask.shape)
                    mask = mask[
                        off_w:off_w+self.crop_size[0],
                        off_h:off_h+self.crop_size[1]]

            if self.task == 'classification':
                return image, label
            else:
                # print("id {} -- image {} -- mask -- {}".format(
                #     idx, image.shape, mask.shape))

                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                # ax = axes.ravel()
                # ax[0].imshow(image.numpy().squeeze()*255, cmap=plt.cm.gray)
                # ax[1].imshow(mask.numpy(), cmap=plt.cm.gray)
                # ax[0].axis('off')
                # ax[1].axis('off')
                # fig.tight_layout()
                # plt.show()

                return image, mask

        if self.training:
            image, gt = _augmentation(
                image, mask, self.input_size,
                flip=True, scale=None,
                crop_size=self.input_size
            )
        else:
            image, gt = _augmentation(
                image, mask, self.input_size,
                flip=False, scale=None,
                crop_size=self.input_size
            )

        return image, gt

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        gt = [item[1] for item in batch]
        images = torch.stack(images, dim=0)
        # labels = torch.LongTensor(labels)
        gt = torch.stack(gt, dim=0)

        return [images, gt]

    def __str__(self):
        return 'Number of samples: {0}\nImages list: '\
            '{1}\nLabels list {2}'.format(
                self.__len__(), self.images, self.labels)

    def get_image_with_bbox(self, idx, linewidth=3):
        if self.images[idx].rsplit('.')[1].lower() == 'dcm':
            ds = pydicom.dcmread(os.path.join(
                self.data_images, self.images[idx]))
            m = ('I;16' if self.bits_per_channel == 16 else 'L') \
                if self.mode == 'L' else 'RGB'
            image = Image.frombuffer(
                m, (ds.Columns, ds.Rows), ds.PixelData, 'raw', m, 0, 1)
        else:
            image = Image.open(
                os.path.join(self.data_images, self.images[idx])
            ).convert(self.mode)
        label = self.labels[idx]

        image = np.array(image)
        if self.bits_per_channel == 16:
            image = np.floor_divide(image, 256)
        # vmin = np.min(image)
        # vmax = np.max(image)
        vmin = 5
        vmax = 53
        f = 255.0 / (vmax - vmin)
        image = np.floor(f * (image - vmin))

        image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)

        l = linewidth
        with open(os.path.join(self.data_labels, label)) as f:
            for line in f.readlines():
                words = line.split(' ')
                assert(len(words) == 5)
                # category = words[0]
                xmin = int(words[1])
                xmax = int(words[2])
                ymin = int(words[3])
                ymax = int(words[4])
                image[ymin-l:ymax+l, xmin-l:xmin+l] = [255, 0, 0]
                image[ymin-l:ymax+l, xmax-l:xmax+l] = [255, 0, 0]
                image[ymin-l:ymin+l, xmin:xmax] = [255, 0, 0]
                image[ymax-l:ymax+l, xmin:xmax] = [255, 0, 0]

        return np.uint8(image)  # Image.fromarray(image)

    def num_defects(self):
        ret = 0
        for label in self.labels:
            with open(os.path.join(self.data_labels, label)) as f:
                ret += len(f.readlines())
        return ret


class InfiniteSampler(Sampler):

    def __init__(self, dataset_length, shuffle=True, seed=0):
        self._size = dataset_length
        self._seed = seed
        self._shuffle = shuffle
        pass

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


def show_dataset(ds):

    linewidth = 4  # actually half-linewidth
    rows = 3
    cols = 4
    offset = 545
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols*3, rows*3), constrained_layout=True)
    fig.suptitle('Samples from ADRIC-FAL-SYN-SIMP dataset', fontsize=14)

    ax = axes.ravel()
    for i in range(0, rows):
        for j in range(0, cols):
            k = i*cols+j
            image = ds.get_image_with_bbox(
                i*cols+j+offset, linewidth=linewidth)
            ax[k].imshow(image.squeeze(), cmap=plt.cm.gray)
            ax[k].axis('off')
    plt.show()


if __name__ == '__main__':

    dataset = CustomDataset(
        data_path='../../datasets/ADRIC-XRIS-FAL-SYN-SIMP-2',
        task='segmentation',
        mode='L',
        bits=16,
        subset='images',
        input_size=(768, 768),
    )
    # for it, e in enumerate(dataset.labels):
    #     if e == '124213843.txt':
    #         print(it)
    # image, label = dataset[545]

    # image = dataset.get_image_with_bbox(545, 4)
    # show_dataset(dataset)
    print(dataset.num_defects())
    # print(image)
    # print(label)
    # print("max val: {}, min val: {}".format(
    #     torch.max(image), torch.min(image)))
    # print("mean val: {}, std val: {}".format(
    #     torch.mean(image), torch.std(image)))
