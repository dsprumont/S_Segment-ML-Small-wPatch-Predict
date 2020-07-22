from __future__ import print_function
import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
    # print(mask)
    # print(np.shape(mask))
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    stats = output[2]  # stats = [topleft-x coord, topleft-y coord, w, h]
    stats = [[x1, y1, x1+w, y1+h, w*h] for x1, y1, w, h, area in stats]
    stats = sorted(stats, key=lambda x: x[4])
    # print('STATS: {}'.format(stats[:-1]))
    return stats[:-1]


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two list of boxes.

    Compute also Intersection-over-Area where Area is only the area of
    the boxes. This can be used in patch-based method where one need to
    decide if part of an object must be kept in the patch.
    TODO: decide if we keep this in the same function or if we create
    multiple version for each.

    Params:
        - boxes1: list, a given list of boxes (eg. prediction boxes)
        - boxes2: list, an other list of boxes (eg. gt boxes)

    Returns:
        - iou: numpy.ndarray, the IoU matrix
        - ioarea1: numpy.ndarray, Intersection-over-Area1 matrix
        - ioarea2: numpy.ndarray, Intersection-over-Area2 matrix
    """
    b1 = np.array(boxes1, dtype=np.int32)  # M x 5
    b2 = np.array(boxes2, dtype=np.int32)  # N x 5

    min_xy = np.minimum(b1[:, None, 2:4], b2[:, 2:4])
    max_xy = np.maximum(b1[:, None, :2], b2[:, :2])
    wh = min_xy-max_xy
    wh = np.clip(wh, 0, None)
    inter_area = np.prod(wh, axis=2)  # M x N
    area2 = b2[:, 4]

    iou = np.where(
        inter_area > 0,
        inter_area / (b1[:, None, 4] + area2 - inter_area),
        0
    )
    ioarea1 = np.where(
        inter_area > 0,
        inter_area / b1[:, None, 4],
        0
    )
    ioarea2 = np.where(
        inter_area > 0,
        inter_area / area2,
        0
    )
    return iou, ioarea1, ioarea2


def compute_iou_matrix_with_contours(
        contours1, contours2, dimensions=(256, 256)):
    """
    Computes IoU matrix between two list of contours.
    This methods use cv2.fillPoly to build individual mask
    and logical and/or operation to compute intersection/union.

    Params:
        - contours1: list, a list of contours provided by cv2.findContours
        - contours2: list, a list of contours provided by cv2.findContours
        - dimensions: tuple, a tuple of (H, W) dimensions of the image/patch

    Returns:
        - iou_mat: np.array, the IoU matrix between the list of contours
    """
    n_c1 = np.size(contours1, 0)
    n_c2 = np.size(contours2, 0)
    iou_mat = np.zeros((n_c1, n_c2))
    blank = np.zeros(dimensions)
    for c1, contour1 in enumerate(contours1):
        blank1 = cv2.fillPoly(blank.copy(), [contour1], (1))
        for c2, contour2 in enumerate(contours2):
            blank2 = cv2.fillPoly(blank.copy(), [contour2], (1))
            intersection = np.sum(np.logical_and(blank1, blank2))
            areas = np.sum(np.logical_or(blank1, blank2))
            iou_mat[c1, c2] = intersection / (areas + 1e-10)
    return iou_mat


def evaluation_on_segmentation(model, dataloader, iou_thresh=0.5):
    """
    Evaluate segmentation performance on detection task using
    connected components analysis. WARNING: this method does not work with
    patch-based dataset.

    Use `torch.argmax` to compute the predicted pixel value.
    Params:
        - model: torch.nn.Module, a model object
        - dataloader: torch.utils.data.Dataloader, a dataloader object
        - iou_thresh: float, thresh defining whether predicted and ground
        thruth bounding boxes target the same object or not.

    Returns:
        - stats (numpy.ndarray of float): an array with columns:
            p: the number of positive samples
            tp: the number of true positive predictions
            tn: the number of true negative predictions (set to 0)
            fp: the number of false positive predictions
            fn: the number of false negative predictions
        for each image (row) in the dataset
        - dataset_stats: (numpy.ndarray of float): sum of each stats
        over the dataset
    """
    _gts_boxes = []
    _preds_boxes = []
    _stats = []  # p tp, tn, fp, fn

    softmax = nn.Softmax(dim=1)

    local_time = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{local_time}] Evaluation of model on segmentation task:')
    with eval_context(model), torch.no_grad():
        # accumulate predictions over the dataset
        max_len = len(dataloader.dataset)
        count = 0
        for it, batch in enumerate(dataloader):
            images = batch[0].cuda()
            groundtruths = batch[1].numpy()

            predictions = model.forward(images)['logits']
            predictions = torch.argmax(softmax(predictions), dim=1)
            predictions = predictions.cpu().numpy()

            for e in range(groundtruths.shape[0]):
                gt_box = compute_box_from_mask(np.uint8(groundtruths[0]))
                pred_box = compute_box_from_mask(np.uint8(predictions[0]))
                _gts_boxes.append(gt_box)
                _preds_boxes.append(pred_box)
            count += groundtruths.shape[0]

            if (it+1) % 20 == 0 or count == max_len:
                local_time = time.strftime("%H:%M:%S", time.localtime())
                print(f'[{local_time}] Processing batch {it:3d}\t'
                      f'{count}/{max_len} images have already been processed')

        # compute iou btw gt and pred for each image
        for it in range(len(_gts_boxes)):
            gt_box = _gts_boxes[it]
            pred_box = _preds_boxes[it]

            # no instances in gt or pred
            if len(gt_box) + len(pred_box) == 0:
                _stats.append([0, 0, 0, 0, 0])
            # instances in pred but not in gt
            elif len(gt_box) == 0:
                _stats.append([0, 0, 0, len(pred_box), 0])
            # instances in gt but not in pred
            elif len(pred_box) == 0:
                _stats.append([len(gt_box), 0, 0, 0, len(gt_box)])
            # instances in both gt and pred
            else:
                iou, _, _ = compute_iou_matrix(gt_box, pred_box)
                tps = iou >= iou_thresh
                tp = np.sum(tps)
                fp = np.count_nonzero(np.sum(tps, axis=0) == 0)
                fn = np.count_nonzero(np.sum(tps, axis=1) == 0)
                _stats.append([len(gt_box), tp, 0, fp, fn])

            if (it+1) % 20 == 0 or (it+1) == max_len:
                local_time = time.strftime("%H:%M:%S", time.localtime())
                print(f'[{local_time}] Computing statistics..'
                      f'{(it+1)}/{max_len} images have already been processed')

    # Statistics for each images can now be aggregated
    stats = np.array(_stats)
    dataset_stats = np.sum(stats, axis=0)

    # print statistics
    local_time = time.strftime("%H:%M:%S", time.localtime())
    pre = int(dataset_stats[1]*100/(dataset_stats[1] + dataset_stats[3]))
    print(f'[{local_time}] Statistics:\n'
          f'Positive: {dataset_stats[0]}\n'
          f'True Positive: {dataset_stats[1]}\n'
          f'True Negative: {dataset_stats[2]}\n'
          f'False Positive: {dataset_stats[3]}\n'
          f'False Negative: {dataset_stats[4]}\n'
          f'Recall: {int(dataset_stats[1]*100/dataset_stats[0])}%\n'
          f'Precision: {pre}%')

    return stats, dataset_stats


def evaluation_on_segmentation_2(model, dataloader, p_thresh, iou_thresh):
    """
    Evaluate segmentation performance on detection task using
    connected components analysis.

    Use `torch.where(p[cls==1] > p_thresh, 1, 0)` to compute the predicted
    pixel value.

    Params:
        - model (torch.nn.Module): a model object
        - dataloader (torch.utils.data.Dataloader): a dataloader object
        - p_thresh (list[float], float): threshold defining whether the
        probability of a given pixel predicts a positive class or not.
        - iou_thresh (list[float], float): threshold defining whether predicted
        and ground thruth bounding boxes target the same object or not.

    Returns:
        - results (dict) with keys:
            - iou (list[float]): a list with given iou_threshold
            - p (list[float]): a list with given p_threshold
            - stats (list[list[list[float]]]): nested lists of statistics
            for each iou, for each p, gives:
                - positives
                - true positives
                - false positives
                - false negatives
            from which rates, precision and recall can be computed.
    """
    if isinstance(p_thresh, (float,)):
        p_t = [p_thresh]
    elif isinstance(p_thresh, (list,)) and\
            all([isinstance(e, (float,)) for e in p_thresh]):
        p_t = p_thresh
    else:
        raise ValueError("p_thresh expected to be float or "
                         "list[float], got {} instead".format(p_thresh))
    if isinstance(iou_thresh, (float,)):
        iou_t = [iou_thresh]
    elif isinstance(iou_thresh, (list,)) and\
            all([isinstance(e, (float,)) for e in iou_thresh]):
        iou_t = iou_thresh
    else:
        raise ValueError("iou_thresh expected to be float or "
                         "list[float], got {} instead".format(iou_thresh))
    if min(p_t) < 0 or max(p_t) > 1:
        raise ValueError("p_thresh values must be between 0 "
                         "and 1, got {}".format(p_t))
    if min(iou_t) < 0 or max(iou_t) > 1:
        raise ValueError("iou_thresh values must be between 0 "
                         "and 1, got {}".format(iou_t))

    results = {'iou': iou_t, 'p': p_t}
    stats = []
    open_kernel = np.ones((4, 4), np.uint8)
    softmax = nn.Softmax(dim=1)
    with eval_context(model), torch.no_grad():
        for iou_th in iou_t:
            p_stats = []
            for p_th in p_t:
                _gts_boxes = []
                _preds_boxes = []
                _stats = []  # p tp, fp, fn
                max_len = len(dataloader.dataset)
                count = 0
                for it, batch in enumerate(dataloader):
                    images = batch[0].cuda()
                    groundtruths = batch[1].numpy()
                    predictions = model.forward(images)['logits']
                    predictions_1 = softmax(predictions)[:, 1, :, :]
                    # prob = torch.full_like(predictions_1, p)
                    one = torch.ones_like(predictions_1)
                    zero = torch.zeros_like(predictions_1)
                    predictions = torch.where(
                        predictions_1 >= p_th, one, zero)
                    predictions = predictions.cpu().numpy()

                    predictions[groundtruths == 255] = 0
                    groundtruths[groundtruths == 255] = 0

                    # if np.sum(predictions) > 0:
                    #     print("Patch {} has pixel "
                    #           "predictions of 1".format(it))

                    predictions = cv2.morphologyEx(
                        predictions, cv2.MORPH_OPEN, open_kernel,
                        borderType=cv2.BORDER_REPLICATE)

                    for e in range(groundtruths.shape[0]):
                        gt_box = compute_box_from_mask(
                            np.uint8(groundtruths[0]))
                        pred_box = compute_box_from_mask(
                            np.uint8(predictions[0]))
                        _gts_boxes.append(gt_box)
                        _preds_boxes.append(pred_box)
                    count += groundtruths.shape[0]

                    if (it+1) % 100 == 0 or count == max_len:
                        local_time = time.strftime(
                            "%H:%M:%S", time.localtime())
                        print(f'[{local_time}] P_thresh {p_th:2.2f}\t'
                              f'IoU_thresh {iou_th:2.2f}\t'
                              f'Processing batch {it:3d}..\t'
                              f'{count}/{max_len} images have already '
                              f'been processed')

                for it in range(len(_gts_boxes)):
                    gt_box = _gts_boxes[it]
                    pred_box = _preds_boxes[it]

                    # no instances in gt or pred
                    if len(gt_box) + len(pred_box) == 0:
                        _stats.append([0, 0, 0, 0])
                    # instances in pred but not in gt
                    elif len(gt_box) == 0 and len(pred_box) > 0:
                        _stats.append([0, 0, len(pred_box), 0])
                        # print("Patch {} contains false positive".format(
                        #         it))
                        # exit()
                    # instances in gt but not in pred
                    elif len(pred_box) == 0 and len(gt_box) > 0:
                        _stats.append(
                            [len(gt_box), 0, 0, len(gt_box)])
                    # instances in both gt and pred
                    else:
                        iou, _, _ = compute_iou_matrix(
                            gt_box, pred_box)
                        tps = iou >= iou_th
                        # tp = np.sum(tps)
                        tp = np.count_nonzero(np.sum(tps, axis=1) > 0)
                        fp = np.count_nonzero(np.sum(tps, axis=0) == 0)
                        fn = np.count_nonzero(np.sum(tps, axis=1) == 0)
                        _stats.append([len(gt_box), tp, fp, fn])
                        # if fp > 0:
                        #     print("Patch {} contains false positive".format(
                        #         it))
                        #     exit()

                    if (it+1) % 100 == 0 or (it+1) == max_len:
                        local_time = time.strftime(
                            "%H:%M:%S", time.localtime())
                        print(f'[{local_time}] P_thresh {p_th:2.2f}\t'
                              f'IoU_thresh {iou_th:2.2f}\t'
                              f'Computing statistics..\t'
                              f'{(it+1)}/{max_len} images have '
                              f'already been processed')

                # Statistics for each images can now be aggregated
                _stats = np.array(_stats)
                dataset_stats = np.sum(_stats, axis=0)
                p_stats.append(dataset_stats.tolist())

                # print(dataset_stats)
                # print statistics
                local_time = time.strftime("%H:%M:%S", time.localtime())
                prec = int(dataset_stats[1]*100 /
                           (dataset_stats[1] + dataset_stats[2] + 1e-10))
                rec = int(dataset_stats[1]*100/(dataset_stats[0] + 1e-10))
                fnr = int(100*dataset_stats[3]/(dataset_stats[0] + 1e-10))
                fpr = int(100*dataset_stats[2]/(dataset_stats[0] + 1e-10))
                print(f'[{local_time}] Statistics for P threshold of '
                      f'{p_th:2.2f} and ioU Threshold of {iou_th:2.2f}:\n'
                      f'Positive: {dataset_stats[0]}\n'
                      f'True Positive: {dataset_stats[1]}\n'
                      f'False Positive: {dataset_stats[2]} '
                      f'(FPR = {fpr}%)\n'
                      f'False Negative: {dataset_stats[3]} '
                      f'(FNR = {fnr}%)\n'
                      f'Recall: {rec}%\n'
                      f'Precision: {prec}%')

            stats.append(p_stats)
            results['stats'] = stats

    return results


def evaluation_on_segmentation_3(model, dataloader, p_thresh, iou_thresh):
    """
    Evaluate segmentation performance on detection task using
    connected components analysis. Compute additional statistics at image
    level.

    Use `torch.where(p[cls==1] > p_thresh, 1, 0)` to compute the predicted
    pixel value.

    Params:
        - model (torch.nn.Module): a model object
        - dataloader (torch.utils.data.Dataloader): a dataloader object
        - p_thresh (list[float], float): threshold defining whether the
        probability of a given pixel predicts a positive class or not.
        - iou_thresh (list[float], float): threshold defining whether predicted
        and ground thruth bounding boxes target the same object or not.

    Returns:
        - results (dict) with keys:
            - iou (list[float]): a list with given iou_threshold
            - p (list[float]): a list with given p_threshold
            - stats (list[list[list[float]]]): nested lists of statistics
            for each iou, for each p, gives:
                - positives
                - true positives
                - false positives
                - false negatives
            from which rates, precision and recall can be computed.
            - istats (list[list[list[float]]]): nested lists of statistics
            for each iou, for each p, gives:
                - nb. of images with groundthruth defects
                - nb. of images with true postive defects
                - nb. of images with only false positive defects
                - nb. of images with false negative.
            from which rates, precision and recall can be computed.
    """
    if isinstance(p_thresh, (float,)):
        p_t = [p_thresh]
    elif isinstance(p_thresh, (list,)) and\
            all([isinstance(e, (float,)) for e in p_thresh]):
        p_t = p_thresh
    else:
        raise ValueError("p_thresh expected to be float or "
                         "list[float], got {} instead".format(p_thresh))
    if isinstance(iou_thresh, (float,)):
        iou_t = [iou_thresh]
    elif isinstance(iou_thresh, (list,)) and\
            all([isinstance(e, (float,)) for e in iou_thresh]):
        iou_t = iou_thresh
    else:
        raise ValueError("iou_thresh expected to be float or "
                         "list[float], got {} instead".format(iou_thresh))
    if min(p_t) < 0 or max(p_t) > 1:
        raise ValueError("p_thresh values must be between 0 "
                         "and 1, got {}".format(p_t))
    if min(iou_t) < 0 or max(iou_t) > 1:
        raise ValueError("iou_thresh values must be between 0 "
                         "and 1, got {}".format(iou_t))

    results = {'iou': iou_t, 'p': p_t}
    stats = []
    istats = []
    open_kernel = np.ones((4, 4), np.uint8)
    softmax = nn.Softmax(dim=1)
    with eval_context(model), torch.no_grad():
        for iou_th in iou_t:
            p_stats = []
            p_istats = []
            for p_th in p_t:
                _gts_boxes = []
                _preds_boxes = []
                _stats = []  # p tp, fp, fn
                _istats = []
                max_len = len(dataloader.dataset)
                count = 0  # keep track of how many patches have been processed
                im_count = 0  # keep track of how many img. have been processed
                image_id = 0    # id of the image the last patches belong to
                partials = []   # last patches of the partial image image_id
                partial_info = []  # here are the patches metainfo
                partial_gts = []  # corresponding gts
                for it, batch in enumerate(dataloader):
                    patches = batch[0].cuda()
                    gt_patches = batch[1].cuda()
                    meta_info = batch[2]  # [tl_x, tl_y, br_x, br_y, area, id]
                    pred_patches = model.forward(patches)['logits']
                    pred_patches_1 = softmax(pred_patches)[:, 1, :, :]

                    count += patches.shape[0]
                    for m in range(len(meta_info)):
                        if meta_info[m][5] == image_id:
                            # add to the partial list
                            partials.append(pred_patches_1[m, ...])
                            partial_info.append(meta_info[m])
                            partial_gts.append(gt_patches[m])
                        elif meta_info[m][5] == image_id+1:
                            # process all patches for image_id
                            predictions, groundtruth, msk = \
                                _process_patches(
                                    partials, partial_gts, partial_info, p_th)

                            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            # ax = axes.ravel()
                            # ax[0].imshow(predictions, cmap=plt.cm.gray)
                            # ax[1].imshow(groundtruth, cmap=plt.cm.gray)
                            # ax[2].imshow(msk, cmap=plt.cm.gray)
                            # for a in ax:
                            #     a.axis('off')
                            # plt.show()

                            predictions = cv2.morphologyEx(
                                predictions, cv2.MORPH_OPEN, open_kernel,
                                borderType=cv2.BORDER_REPLICATE)
                            _gts_boxes.append(
                                compute_box_from_mask(np.uint8(groundtruth)))
                            _preds_boxes.append(
                                compute_box_from_mask(np.uint8(predictions)))
                            # reset and add to the partial list
                            partials = [pred_patches_1[m, ...]]
                            partial_info = [meta_info[m]]
                            partial_gts = [gt_patches[m]]
                            image_id = image_id + 1
                        else:
                            # this should not happen!
                            raise ValueError(
                                "Patches must be processed from consecutive"
                                " images, got new id != old id + 1")

                    # handle the last image
                    if count == max_len:
                        predictions, groundtruth, _ = \
                            _process_patches(
                                partials, partial_gts, partial_info, p_th)
                        predictions = cv2.morphologyEx(
                            predictions, cv2.MORPH_OPEN, open_kernel,
                            borderType=cv2.BORDER_REPLICATE)
                        _gts_boxes.append(
                            compute_box_from_mask(np.uint8(groundtruth)))
                        _preds_boxes.append(
                            compute_box_from_mask(np.uint8(predictions)))

                    # some logs
                    if (it+1) % 100 == 0 or count == max_len:
                        im_count = image_id+1 if count == max_len else image_id
                        local_time = time.strftime(
                            "%H:%M:%S", time.localtime())
                        print(f'[{local_time}] P_thresh {p_th:2.2f}\t'
                              f'IoU_thresh {iou_th:2.2f}\t'
                              f'Processing batch {it:3d}..\t'
                              f'{count}/{max_len} patches '
                              f'({im_count} images) have already '
                              f'been processed')

                for it in range(len(_gts_boxes)):
                    gt_box = _gts_boxes[it]
                    pred_box = _preds_boxes[it]

                    if len(gt_box) + len(pred_box) == 0:
                        _stats.append([0, 0, 0, 0])
                        _istats.append([0, 0, 0, 0])
                    elif len(gt_box) == 0 and len(pred_box) > 0:
                        _stats.append([0, 0, len(pred_box), 0])
                        _istats.append([0, 0, 1, 0])
                    elif len(pred_box) == 0 and len(gt_box) > 0:
                        _stats.append(
                            [len(gt_box), 0, 0, len(gt_box)])
                        _istats.append([1, 0, 0, 1])
                    else:
                        iou, _, _ = compute_iou_matrix(
                            gt_box, pred_box)
                        tps = iou >= iou_th

                        tp = np.count_nonzero(np.sum(tps, axis=1) > 0)
                        fp = np.count_nonzero(np.sum(tps, axis=0) == 0)
                        fn = np.count_nonzero(np.sum(tps, axis=1) == 0)
                        _stats.append([len(gt_box), tp, fp, fn])
                        if tp > 0:
                            _istats.append([1, 1, 0, 0])
                        else:
                            _istats.append([1, 0, 0, 1])

                    if (it+1) % 20 == 0 or (it+1) == max_len:
                        local_time = time.strftime(
                            "%H:%M:%S", time.localtime())
                        print(f'[{local_time}] P_thresh {p_th:2.2f}\t'
                              f'IoU_thresh {iou_th:2.2f}\t'
                              f'Computing statistics..\t'
                              f'{(it+1)}/{im_count} images have '
                              f'already been processed')

                # Statistics for each images can now be aggregated
                _stats = np.array(_stats)
                dataset_stats = np.sum(_stats, axis=0)
                image_stats = np.sum(_istats, axis=0)
                p_stats.append(dataset_stats.tolist())
                p_istats.append(image_stats.tolist())

                # print(dataset_stats)
                # print statistics
                local_time = time.strftime("%H:%M:%S", time.localtime())
                prec = int(dataset_stats[1]*100 /
                           (dataset_stats[1] + dataset_stats[2] + 1e-10))
                rec = int(dataset_stats[1]*100/(dataset_stats[0] + 1e-10))
                fnr = int(100*dataset_stats[3]/(dataset_stats[0] + 1e-10))
                fpr = int(100*dataset_stats[2]/(dataset_stats[0] + 1e-10))
                iprec = int(image_stats[1]*100 /
                            (image_stats[1] + image_stats[2] + 1e-10))
                irec = int(image_stats[1]*100/(image_stats[0] + 1e-10))
                ifnr = int(100*image_stats[3]/(image_stats[0] + 1e-10))
                ifpr = int(100*image_stats[2]/(image_stats[0] + 1e-10))
                print(f'[{local_time}] Statistics for P threshold of '
                      f'{p_th:2.2f} and ioU Threshold of {iou_th:2.2f}:\n'
                      f'--- defect instances metrics ---\n'
                      f'Positive: {dataset_stats[0]}\n'
                      f'True Positive: {dataset_stats[1]}\n'
                      f'False Positive: {dataset_stats[2]} '
                      f'(FPR = {fpr}%)\n'
                      f'False Negative: {dataset_stats[3]} '
                      f'(FNR = {fnr}%)\n'
                      f'Recall: {rec}%\n'
                      f'Precision: {prec}%\n'
                      f'--- defective images metrics ---\n'
                      f'Images with defects: {image_stats[0]}\n'
                      f'Images with True Positives: {image_stats[1]}\n'
                      f'Images with False Negatives: {image_stats[3]} '
                      f'(FNR = {ifnr}%)\n'
                      f'Images with only False Positives: {image_stats[2]} '
                      f'(FPR = {ifpr}%)\n'
                      f'Recall: {irec}%\n'
                      f'Precision: {iprec}%')

            stats.append(p_stats)
            istats.append(p_istats)
            results['stats'] = stats
            results['istats'] = istats

    return results


def evaluation_on_segmentation_4(model, dataloader, p_thresh, iou_thresh):
    """
    Evaluate segmentation performance on detection task using
    connected components analysis. This method computes Contours instead of
    Boxes to evaluate IoU and statistics. Compute additional statistics at
    image level.

    Use `torch.where(p[cls==1] > p_thresh, 1, 0)` to compute the predicted
    pixel value.

    Params:
        - model (torch.nn.Module): a model object
        - dataloader (torch.utils.data.Dataloader): a dataloader object
        - p_thresh (list[float], float): threshold defining whether the
        probability of a given pixel predicts a positive class or not.
        - iou_thresh (list[float], float): threshold defining whether predicted
        and ground thruth bounding boxes target the same object or not.

    Returns:
        - results (dict) with keys:
            - iou (list[float]): a list with given iou_threshold
            - p (list[float]): a list with given p_threshold
            - stats (list[list[list[float]]]): nested lists of statistics
            for each iou, for each p, gives:
                - positives
                - true positives
                - false positives
                - false negatives
            from which rates, precision and recall can be computed.
            - istats (list[list[list[float]]]): nested lists of statistics
            for each iou, for each p, gives:
                - nb. of images with groundthruth defects
                - nb. of images with true postive defects
                - nb. of images with only false positive defects
                - nb. of images with false negative.
            from which rates, precision and recall can be computed.
    """
    if isinstance(p_thresh, (float,)):
        p_t = [p_thresh]
    elif isinstance(p_thresh, (list,)) and\
            all([isinstance(e, (float,)) for e in p_thresh]):
        p_t = p_thresh
    else:
        raise ValueError("p_thresh expected to be float or "
                         "list[float], got {} instead".format(p_thresh))
    if isinstance(iou_thresh, (float,)):
        iou_t = [iou_thresh]
    elif isinstance(iou_thresh, (list,)) and\
            all([isinstance(e, (float,)) for e in iou_thresh]):
        iou_t = iou_thresh
    else:
        raise ValueError("iou_thresh expected to be float or "
                         "list[float], got {} instead".format(iou_thresh))
    if min(p_t) < 0 or max(p_t) > 1:
        raise ValueError("p_thresh values must be between 0 "
                         "and 1, got {}".format(p_t))
    if min(iou_t) < 0 or max(iou_t) > 1:
        raise ValueError("iou_thresh values must be between 0 "
                         "and 1, got {}".format(iou_t))

    results = {'iou': iou_t, 'p': p_t}
    stats = []
    istats = []
    open_kernel = np.ones((4, 4), np.uint8)
    softmax = nn.Softmax(dim=1)
    with eval_context(model), torch.no_grad():
        for iou_th in iou_t:
            p_stats = []
            p_istats = []
            for p_th in p_t:
                _stats = []  # p tp, fp, fn
                _istats = []
                max_len = len(dataloader.dataset)
                count = 0  # keep track of how many patches have been processed
                im_count = 0  # keep track of how many img. have been processed
                image_id = 0    # id of the image the last patches belong to
                partials = []   # last patches of the partial image image_id
                partial_info = []  # here are the patches metainfo
                partial_gts = []  # corresponding gts
                for it, batch in enumerate(dataloader):
                    patches = batch[0].cuda()
                    gt_patches = batch[1].cuda()
                    meta_info = batch[2]  # [tl_x, tl_y, br_x, br_y, area, id]
                    pred_patches = model.forward(patches)['logits']
                    pred_patches_1 = softmax(pred_patches)[:, 1, :, :]

                    count += patches.shape[0]
                    for m in range(len(meta_info)):
                        if meta_info[m][5] == image_id:
                            # add to the partial list
                            partials.append(pred_patches_1[m, ...])
                            partial_info.append(meta_info[m])
                            partial_gts.append(gt_patches[m])
                        elif meta_info[m][5] == image_id+1:
                            # process all patches for image_id
                            predictions, groundtruth, _ = \
                                _process_patches(
                                    partials, partial_gts, partial_info, p_th)
                            predictions = cv2.morphologyEx(
                                predictions, cv2.MORPH_OPEN, open_kernel,
                                borderType=cv2.BORDER_REPLICATE)

                            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            # ax = axes.ravel()
                            # ax[0].imshow(predictions, cmap=plt.cm.gray)
                            # ax[1].imshow(groundtruth, cmap=plt.cm.gray)
                            # ax[2].imshow(msk, cmap=plt.cm.gray)
                            # for a in ax:
                            #     a.axis('off')
                            # plt.show()

                            # compute image-level statistics
                            a, b = _get_stats_from_masks(
                                predictions,    # binary array of predictions
                                groundtruth,    # binary gt mask
                                iou_th          # IoU threshold
                            )
                            _stats.append(a)
                            _istats.append(b)

                            # reset and add to the partial list
                            partials = [pred_patches_1[m, ...]]
                            partial_info = [meta_info[m]]
                            partial_gts = [gt_patches[m]]
                            image_id = image_id + 1
                        else:
                            # this should not happen!
                            raise ValueError(
                                "Patches must be processed from consecutive"
                                " images, got new id != old id + 1")

                    # handle the last image
                    if count == max_len:
                        predictions, groundtruth, _ = \
                            _process_patches(
                                partials, partial_gts, partial_info, p_th)
                        predictions = cv2.morphologyEx(
                            predictions, cv2.MORPH_OPEN, open_kernel,
                            borderType=cv2.BORDER_REPLICATE)

                        # compute image-level statistics
                        a, b = _get_stats_from_masks(
                            predictions,    # binary array of predictions
                            groundtruth,    # binary gt mask
                            iou_th          # IoU threshold
                        )
                        _stats.append(a)
                        _istats.append(b)

                    # some logs
                    if (it+1) % 100 == 0 or count == max_len:
                        im_count = image_id+1 if count == max_len else image_id
                        local_time = time.strftime(
                            "%H:%M:%S", time.localtime())
                        print(f'[{local_time}] P_thresh {p_th:2.2f}\t'
                              f'IoU_thresh {iou_th:2.2f}\t'
                              f'Computing statistics..\t'
                              f'{(it+1)}/{im_count} images have '
                              f'already been processed')

                _stats = np.array(_stats)
                dataset_stats = np.sum(_stats, axis=0)
                image_stats = np.sum(_istats, axis=0)
                p_stats.append(dataset_stats.tolist())
                p_istats.append(image_stats.tolist())

                # print statistics
                local_time = time.strftime("%H:%M:%S", time.localtime())
                prec = int(dataset_stats[1]*100 /
                           (dataset_stats[1] + dataset_stats[2] + 1e-10))
                rec = int(dataset_stats[1]*100/(dataset_stats[0] + 1e-10))
                fnr = int(100*dataset_stats[3]/(dataset_stats[0] + 1e-10))
                fpr = int(100*dataset_stats[2]/(dataset_stats[0] + 1e-10))
                iprec = int(image_stats[1]*100 /
                            (image_stats[1] + image_stats[2] + 1e-10))
                irec = int(image_stats[1]*100/(image_stats[0] + 1e-10))
                ifnr = int(100*image_stats[3]/(image_stats[0] + 1e-10))
                ifpr = int(100*image_stats[2]/(image_stats[0] + 1e-10))
                print(f'[{local_time}] Statistics for P threshold of '
                      f'{p_th:2.2f} and ioU Threshold of {iou_th:2.2f}:\n'
                      f'--- defect instances metrics ---\n'
                      f'Positive: {dataset_stats[0]}\n'
                      f'True Positive: {dataset_stats[1]}\n'
                      f'False Positive: {dataset_stats[2]} '
                      f'(FPR = {fpr}%)\n'
                      f'False Negative: {dataset_stats[3]} '
                      f'(FNR = {fnr}%)\n'
                      f'Recall: {rec}%\n'
                      f'Precision: {prec}%\n'
                      f'--- defective images metrics ---\n'
                      f'Images with defects: {image_stats[0]}\n'
                      f'Images with True Positives: {image_stats[1]}\n'
                      f'Images with False Negatives: {image_stats[3]} '
                      f'(FNR = {ifnr}%)\n'
                      f'Images with only False Positives: {image_stats[2]} '
                      f'(FPR = {ifpr}%)\n'
                      f'Recall: {irec}%\n'
                      f'Precision: {iprec}%')

            stats.append(p_stats)
            istats.append(p_istats)
            results['stats'] = stats
            results['istats'] = istats

    return results


def inference_on_segmentation(model, dataloader, p_thresh):
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
    softmax = nn.Softmax(dim=1)
    with eval_context(model), torch.no_grad():
        max_len = len(dataloader.dataset)
        count = 0  # keep track of how many patches have been processed
        im_count = 0  # keep track of how many img. have been processed
        image_id = 0    # id of the image the last patches belong to
        partials = []   # last patches of the partial image image_id
        partial_info = []  # here are the patches metainfo
        partial_gts = []  # corresponding gts
        for it, batch in enumerate(dataloader):
            patches = batch[0].cuda()
            gt_patches = batch[1].cuda()
            meta_info = batch[2]  # [tl_x, tl_y, br_x, br_y, area, id]
            pred_patches = model.forward(patches)['logits']
            pred_patches_1 = softmax(pred_patches)[:, 1, :, :]

            count += patches.shape[0]
            for m in range(len(meta_info)):
                if meta_info[m][5] == image_id:
                    # add to the partial list
                    partials.append(pred_patches_1[m, ...])
                    partial_info.append(meta_info[m])
                    partial_gts.append(gt_patches[m])
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
                    partial_gts = [gt_patches[m]]
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


def _get_stats_from_masks(predictions, groundtruth, iou_threshold):
    """
    Compute statistics for defects and defective images.

    Params:
        - predictions (numpy.darray), an array with probability for each pixel
        - groundtruth (numpy.darray), an array with ground-truth annotation
        - iou_threshold (float), the treshold under wihch pixels are labelled
          background

    Returns:
        - results (tuple) made of:
            - stats (list[int]), list of P, TP, FP, FN for all defect in the
              given image
            - istats (list[int]), list containing 1 or 0 for the following
              categories:
                - does this image contain defects ? 1: Yes, 0: No
                - does the model predict TP ?
                - does the model predict only FP ?
                - does the model predict FN ?
    """
    pd_contours, _ = cv2.findContours(
        predictions,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    gt_contours, _ = cv2.findContours(
        groundtruth,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if not (pd_contours and gt_contours):
        # no prediction and no gt
        return ([0, 0, 0, 0], [0, 0, 0, 0])
    elif not gt_contours and pd_contours:
        # prediction but no gt -> False Positive
        return ([0, 0, len(pd_contours), 0], [0, 0, 1, 0])
    elif gt_contours and not pd_contours:
        # no prediction but gt -> False Negative
        return ([len(gt_contours), 0, 0, len(gt_contours)], [1, 0, 0, 1])
    else:
        # predictions and gts
        iou = compute_iou_matrix_with_contours(
            gt_contours,
            pd_contours,
            np.size(predictions)
        )
        tps = iou >= iou_threshold
        tp = np.count_nonzero(np.sum(tps, axis=1) > 0)
        fp = np.count_nonzero(np.sum(tps, axis=0) == 0)
        fn = np.count_nonzero(np.sum(tps, axis=1) == 0)

        return ([len(gt_contours), tp, fp, fn],
                [1, 1, 0, 0] if tp > 0 else [1, 0, 0, 1])


def _process_patches(patches, patches_gts, patches_info, p_threshold):
    """
    Reconstruct image-level gt mask and prediction mask from patches.
    Each image-level pixel element is the average value of all patches' pixels
    corresponding to this image-level pixel.
    """
    # get the display of the patches into the image
    np_patches_info = np.array(patches_info)
    height, width = np.max(np_patches_info[:, 2:4], axis=0)
    width += 1
    height += 1

    # reconstruct the image prediction from patches predictions
    p_gth = torch.zeros((width, height), dtype=torch.float).cuda()
    p_msk = torch.zeros((width, height), dtype=torch.float).cuda()
    p_img = torch.zeros((width, height), dtype=torch.float).cuda()
    zeros = torch.zeros_like(patches[0], dtype=torch.float).cuda()
    for patch, gt, mi in zip(patches, patches_gts, patches_info):

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # ax = axes.ravel()
        # ax[0].imshow(patch.cpu().numpy(), cmap=plt.cm.gray)
        # ax[1].imshow(gt.cpu().numpy(), cmap=plt.cm.gray)
        # for a in ax:
        #     a.axis('off')
        # plt.show()
        gt = gt.float()
        clean_patch = torch.where(gt == 255, zeros, patch)
        p_img[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += clean_patch
        p_msk[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += torch.ones_like(
            patch, dtype=torch.float)
        clean_gt = torch.where(gt == 255, zeros, gt)
        p_gth[mi[1]:mi[3]+1, mi[0]:mi[2]+1] += clean_gt
    p_img.div_(p_msk)  # linear average
    p_gth.div_(p_msk)  # linear average

    # apply the probability threshold to the image avg softmax
    one = torch.ones_like(p_img)
    zero = torch.zeros_like(p_img)
    predictions = torch.where(p_img >= p_threshold, one, zero)
    predictions = predictions.cpu().numpy()
    groundtruth = p_gth.cpu().numpy()
    return predictions, groundtruth, p_msk.cpu().numpy()


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

    # reconstruct the image prediction from patches predictions
    p_msk = torch.zeros((width, height), dtype=torch.float).cuda()
    p_img = torch.zeros((width, height), dtype=torch.float).cuda()
    zeros = torch.zeros_like(patches[0], dtype=torch.float).cuda()
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
            # image = io.imread(
            #     os.path.join(data_path, image))
            image = np.array(Image.open(os.path.join(data_path, image)))
        print(image.shape)
        if len(image.shape) > 2:
            means += np.mean(image[:, :, :3], axis=tuple(range(2)))
            stds += np.std(image[:, :, :3], axis=tuple(range(2)))
        else:
            means += np.mean(image)
            stds += np.std(image)
    f = 256 ** (int(bits / 8)) - 1
    means = means / (len(images) * f)
    stds = stds / (len(images) * f)
    return means, stds


def plot_image_with_channels(data_path):
    images = os.listdir(data_path)
    images = [image for image in images
              if image.rsplit('.')[1]
              in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']]

    for image in images:
        # image = io.imread(
        #     os.path.join(data_path, image))
        image = np.array(Image.open(os.path.join(data_path, image)))

        fig, axes = plt.subplots(1, 5, figsize=(20, 20))
        ax = axes.ravel()
        ax[0].imshow(image, cmap=plt.cm.gray if len(image.shape) < 3 else None)
        ax[0].axis('off')
        if len(image.shape) > 2:
            for i in range(image.shape[2]):
                tmp = np.zeros(image.shape, dtype='uint8')
                tmp[:, :, i] = image[:, :, i]
                ax[i+1].imshow(tmp)
                ax[i+1].axis('off')
        fig.tight_layout()
        plt.show()
        break


def dicom_image(path):
    ds = pydicom.dcmread(path)
    print(ds)
    for k, v in ds.items():
        print("key : {} ---> value: {}".format(k, v))
    image = np.frombuffer(ds.PixelData, dtype="uint16")
    return np.reshape(image, (ds.Rows, ds.Columns))
