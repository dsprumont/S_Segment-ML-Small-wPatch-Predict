from __future__ import print_function
import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pydicom
from contextlib import contextmanager


@contextmanager
def eval_context(model):  # inspired from detectron2
    """
    Change the context to eval_mode and restore the previous context at exit.
    """
    trainmode = model.training
    model.eval()
    yield
    model.train(trainmode)


def compute_box_from_mask(mask, connectivity=8, ignore_index=255):
    """
    Identify boxes from a binary mask. Use `cv2.connectedComponentsWithstats`
    analysis.

    Params:
        - mask: numpy.ndarray, a binary mask
        - connectivity: int, choose between 4 (left, right, up, down) or
        8 (4 + diagonals) to connected elements together.
        - ignore_index: int, allows to ignore a special value in the mask.
        (introduce during padding for example)

    Returns:
        - a list of box [top-left x, top-left y, bottom-right x, bottom-right
        y, area]
    """
    mask = np.where(mask != ignore_index, mask, 0)

    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    stats = output[2]  # stats = [topleft-x coord, topleft-y coord, w, h]
    stats = [[x1, y1, x1+w, y1+h, w*h] for x1, y1, w, h, area in stats]
    stats = sorted(stats, key=lambda x: x[4])

    return stats[:-1]


def inference_on_segmentation(model, dataloader, p_thresh, device="cpu"):
    """
    Perform inference on detection task using segmentation model and
    connected components analysis.

    Use `torch.where(p[cls==1] > p_thresh, 1, 0)` to compute the predicted
    pixel value.

    Params:
        - model (torch.nn.Module): a model object.
        - dataloader (torch.utils.data.Dataloader): a dataloader object.
        - p_thresh ( float): threshold defining whether the
        probability of a given pixel predicts a positive class or not.
        - device (torch.device): the device used to execute pytorch code.
        Can be either a torch.device or a string like 'cpu' or 'cuda:0'.

    Returns:
        - results (list), list of instance. Each instance is a dict
        with key:
            - image_id (int), the id of the image in the dataset
            - bbox (list), a list of [x1 y1 x2 y2 area] boxes
    """
    if isinstance(p_thresh, (float,)):
        p_t = p_thresh
    elif isinstance(p_thresh, (list,)) and isinstance(p_thresh[0], (float,)):
        p_t = p_thresh[0]
    else:
        raise ValueError("p_thresh expected to be float, "
                         "got {} instead".format(p_thresh))
    if p_t < 0 or p_t > 1:
        raise ValueError("p_thresh value must be between 0 "
                         "and 1, got {}".format(p_t))

    _preds_boxes = []
    open_kernel = np.ones((4, 4), np.uint8)
    softmax = nn.Softmax(dim=1).to(device)
    with eval_context(model), torch.no_grad():
        max_len = len(dataloader.dataset)
        count = 0  # keep track of how many patches have been processed
        im_count = 0  # keep track of how many img. have been processed
        image_id = 0    # id of the image the last patches belong to
        partials = []   # last patches of the partial image image_id
        partial_info = []  # here are the patches metainfo
        for it, batch in enumerate(dataloader):
            patches = batch[0].to(device)
            meta_info = batch[1]  # [tl_x, tl_y, br_x, br_y, area, id]
            pred_patches = model.forward(patches)['logits']
            pred_patches_1 = softmax(pred_patches)[:, 1, :, :]

            count += patches.shape[0]
            for m in range(len(meta_info)):
                if meta_info[m][5] == image_id:
                    # add to the partial list
                    partials.append(pred_patches_1[m, ...])
                    partial_info.append(meta_info[m])
                elif meta_info[m][5] == image_id+1:
                    # process all patches for image_id
                    predictions = _process_patches_pred_only(
                        partials, None, partial_info, p_t)

                    predictions = cv2.morphologyEx(
                        predictions, cv2.MORPH_OPEN, open_kernel,
                        borderType=cv2.BORDER_REPLICATE)

                    _pred_boxes = {
                        'filename': dataloader.dataset.get_name(image_id),
                        'image_id': image_id,
                        'size': dataloader.dataset.get_size(image_id)
                    }
                    _pred_boxes['bbox'] = compute_box_from_mask(
                        np.uint8(predictions))
                    _preds_boxes.append(_pred_boxes)
                    # reset and add to the partial list
                    partials = [pred_patches_1[m, ...]]
                    partial_info = [meta_info[m]]
                    image_id = image_id + 1
                else:
                    # this should not happen!
                    raise ValueError(
                        "Patches must be processed from consecutive"
                        " images, got new id != old id + 1")

            # handle the last image
            if count == max_len:
                predictions = _process_patches_pred_only(
                    partials, None, partial_info, p_t)
                predictions = cv2.morphologyEx(
                    predictions, cv2.MORPH_OPEN, open_kernel,
                    borderType=cv2.BORDER_REPLICATE)
                _pred_boxes = {
                    'filename': dataloader.dataset.get_name(image_id),
                    'image_id': image_id,
                    'size': dataloader.dataset.get_size(image_id)
                }
                _pred_boxes['bbox'] = compute_box_from_mask(
                    np.uint8(predictions))
                _preds_boxes.append(_pred_boxes)

            # some logs
            if (it+1) % 100 == 0 or count == max_len:
                im_count = image_id+1 if count == max_len else image_id
                local_time = time.strftime(
                    "%H:%M:%S", time.localtime())
                print(f'[{local_time}] P_thresh {p_t:2.2f}\t'
                      f'Processing batch {it:3d}..\t'
                      f'{count}/{max_len} patches '
                      f'({im_count} images) have already '
                      f'been processed')

    return _preds_boxes


def _process_patches_pred_only(
        patches, patches_gts, patches_info, p_threshold):
    """
    Reconstruct image-level gt mask and prediction mask from patches.
    Only reconstruct prediction. Use this version if you already have
    image-level ground-truth segmentation annotation.
    Each image-level pixel element is the average value of all patches' pixels
    corresponding to this image-level pixel.
    """
    # get the display of the patches into the image
    np_patches_info = np.array(patches_info)
    height, width = np.max(np_patches_info[:, 2:4], axis=0)
    width += 1
    height += 1

    _device = patches[0].device
    # reconstruct the image prediction from patches predictions
    p_msk = torch.zeros((width, height), dtype=torch.float).to(_device)
    p_img = torch.zeros((width, height), dtype=torch.float).to(_device)
    zeros = torch.zeros_like(patches[0], dtype=torch.float).to(_device)
    if patches_gts is None:
        for patch, mi in zip(patches, patches_info):
            p_img[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += patch
            p_msk[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += torch.ones_like(
                patch, dtype=torch.float)
    else:
        for patch, gt, mi in zip(patches, patches_gts, patches_info):

            gt = gt.float()
            clean_patch = torch.where(gt == 255, zeros, patch)
            p_img[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += clean_patch
            p_msk[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += torch.ones_like(
                patch, dtype=torch.float)
    p_img.div_(p_msk)  # linear average

    # apply the probability threshold to the image avg softmax
    one = torch.ones_like(p_img)
    zero = torch.zeros_like(p_img)
    predictions = torch.where(p_img >= p_threshold, one, zero)
    predictions = predictions.cpu().numpy()
    return predictions


def compute_mean_and_std(data_path, bits=8):
    """
    Return means and stds as list of 3 values (rgb).

    If input is grayscale then all 3 values are the same.
    """
    images = os.listdir(data_path)
    images = [image for image in images
              if image.rsplit('.')[1].lower()
              in ['jpg', 'jpeg', 'png', 'dcm']]

    means = np.zeros(3)
    stds = np.zeros(3)
    for image in images:
        if image.rsplit('.')[1].lower() == 'dcm':
            ds = pydicom.dcmread(os.path.join(data_path, image))
            image = np.reshape(np.frombuffer(
                ds.PixelData, dtype="uint{}".format(bits)),
                 (ds.Rows, ds.Columns))
        else:
            image = np.array(Image.open(os.path.join(data_path, image)))
        if len(image.shape) > 2:
            means += np.mean(image[:, :, :3], axis=tuple(range(2)))
            stds += np.std(image[:, :, :3], axis=tuple(range(2)))
        else:
            means += np.mean(image)
            stds += np.std(image)
    f = 256 ** (int(bits / 8)) - 1
    means = means / (len(images) * f)
    stds = stds / (len(images) * f)
    return means.tolist(), stds.tolist()
