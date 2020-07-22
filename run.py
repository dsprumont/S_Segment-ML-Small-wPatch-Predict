# common dependencies
from __future__ import print_function
import os
import sys
import json
import argparse
import numpy as np

# deep learning denpendencies
from torch.utils.data import DataLoader

# custom dependencies
from models.fpn import FPN
from data.detection.patch_dataset import PatchBasedDataset
from utils import (
    inference_on_segmentation,
    compute_mean_and_std
)

from shapely.geometry import Point, Polygon
from cytomine import CytomineJob
from cytomine.models import (
    ImageInstanceCollection,
    AnnotationCollection,
    Annotation,
    Job
)


def load_json(file_name):
    """
    Load .json specifed by file_name into a python dict.
    """
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise argparse.ArgumentTypeError("Got error: {}".format(e))


def str2floatlist(string):
    """
    Parse the given string into a list of float.

    Examples:
        0.5                     -> [0.5]
        (0.7,0.8)               -> [0.7, 0.8]
        (0.1:0.5:0.1)           -> [0.1, 0.2, 0.3, 0.4, 0.5]
        (0.1,0.3:0.5:0.05,0.8)  -> [0.1, 0.3, 0.35, 0.40, 0.45, 0.5, 0.8]
    """
    ret = list()
    if string.startswith('(') and string.endswith(')'):
        print(string)
        string = string[1:len(string)-1]
        print(string)
        strings = string.split(',')
        print(strings)
        for string in strings:
            string = string.strip()
            substr = string.split(':')
            if len(substr) < 2:
                try:
                    string = float(substr[0])
                except ValueError as e:
                    raise argparse.ArgumentTypeError(
                        "In {}, got error: {}".format(substr[0], e))
                ret.append(string)
            elif len(substr) == 3:
                try:
                    begin = float(substr[0])
                    end = float(substr[1])
                    step = float(substr[2])
                    vec = np.arange(begin, end, step, dtype=np.float)
                except ValueError as e:
                    raise argparse.ArgumentTypeError(
                        "In {}, got error: {}".format(string, e))
                if len(vec) < 1:
                    raise argparse.ArgumentTypeError(
                        "In {}, got error: {}".format(
                            string, "Step argument > End argument"))
                else:
                    ret.extend(list(vec))
                    ret.append(end)
            else:
                raise argparse.ArgumentTypeError(
                    "In {}, got error: {}".format(string, "Too many elements"))
    else:
        try:
            string = float(string)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                "In {}, got error: {}".format(string, e))
        ret.append(string)
    return ret


def main(argv):
    with CytomineJob.from_cli(argv) as conn:
        conn.job.update(progress=0, statusComment="Initialization..")
        base_path = "{}".format(os.getenv("HOME"))  # Mandatory for Singularity
        working_path = os.path.join(base_path, str(conn.job.id))

        # Load pretrained model (assume the best of all)
        conn.job.update(progress=0,
                        statusComment="Loading segmentation model..")
        with open(os.path.join(
                base_path, "/resnet50b_fpn256/config.json")) as f:
            config = json.load(f)
        model = FPN.build_resnet_fpn(
            name=config['model_name'],
            input_size=conn.parameters.model_patch_size,  # must be / by 16
            input_channels=1 if config['image_mode'] == 'grayscale' else 3,
            output_channels=config['fpn_out_channels'],
            num_classes=config['num_classes'],
            in_features=config['fpn_in_features'],
            out_features=config['fpn_out_features']
        )

        # Select images to process
        images = ImageInstanceCollection().fetch_with_filter(
            "project", conn.parameters.cytomine_id_project)
        list_imgs = []
        if conn.parameters.cytomine_id_images == 'all':
            for image in images:
                list_imgs.append(int(image.id))
        else:
            list_imgs = [int(id_img)
                         for id_img in
                         conn.parameters.cytomine_id_images.split(',')]

        # Download selected images into "working_directory"
        img_path = os.path.join(working_path, "images")
        for image in conn.monitor(
                list_imgs, start=2, end=50, period=0.1,
                prefix="Downloading images into working directory.."):
            fname, fext = os.path.splitext(image.filename)
            if image.download(dest_pattern=os.path.join(
                    img_path, "{}".format(image.filename))) is True:
                os.rename(
                    os.path.join(img_path, "{}".format(image.filename)),
                    os.path.join(img_path, "{}{}".format(image.id, fext))
                )

        # create a file that lists all images (used by PatchBasedDataset
        conn.job.update(progress=50,
                        statusComment="Preparing data for execution..")
        images = os.listdir(img_path)
        images = list(map(lambda x: x+'\n', images))
        with open(os.path.join(working_path), 'w') as f:
            f.writelines(images)

        # Prepare dataset and dataloader objects
        ImgTypeBits = {'.dcm': 16}
        channel_bits = ImgTypeBits.get(fext.lower(), 8)
        mean, std = compute_mean_and_std(working_path, bits=channel_bits)

        dataset = PatchBasedDataset(
            path=working_path,
            subset='images',
            patch_size=conn.parameters.dataset_patch_size,
            mode=config['image_mode'],
            bits=channel_bits,
            mean=mean,
            std=std,
            training=False,
            filter='none'  # This script is meant for inference
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=conn.parameters.model_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            collate_fn=PatchBasedDataset.collate_fn
        )

        # Go over images
        conn.job.update(status=Job.RUNNING, progress=55,
                        statusComment="Running inference on images..")
        results = inference_on_segmentation(
            model, dataloader, conn.parameters.postprocess_p_threshold)

        for id_image in conn.monitor(
                list_imgs, start=90, end=95,
                prefix="Deleting old annotations on images..", period=0.1):
            # Delete old annotations
            del_annotations = AnnotationCollection()
            del_annotations.image = id_image
            del_annotations.user = conn.job.id
            del_annotations.project = conn.parameters.cytomine_id_project
            del_annotations.term = conn.parameters.cytomine_id_predict_term,
            del_annotations.fetch()
            for annotation in del_annotations:
                annotation.delete()

        conn.job.update(
            status=Job.RUNNING, progress=95,
            statusComment="Uploading new annotations to Cytomine server..")
        annotations = AnnotationCollection()
        for instance in results:
            idx, _ = os.path.splitext(instance['filename'])
            width, height = instance['size']

            for box in instance['bbox']:
                points = [
                    Point(box[0], height-1-box[1]),
                    Point(box[0], height-1-box[3]),
                    Point(box[2], height-1-box[3]),
                    Point(box[2], height-1-box[1])
                ]
                annotation = Polygon(points)

                annotations.append(Annotation(
                    location=annotation.wkt,
                    id_image=int(idx),
                    id_terms=conn.parameters.cytomine_id_predict_term,
                    id_project=conn.parameters.cytomine_id_project)
                )
        annotations.save()

        conn.job.update(
            status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == '__main__':
    main(sys.argv[1:])