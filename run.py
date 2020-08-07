# common dependencies
from __future__ import print_function
import os
import sys
import json
from shapely.geometry import Point, Polygon

# deep learning denpendencies
import torch
from torch.utils.data import DataLoader

# cytomine dependencies
from cytomine import CytomineJob
from cytomine.models import (
    ImageInstanceCollection,
    AnnotationCollection,
    Annotation,
    Job
)

# custom dependencies
from models.fpn import FPN
from data.detection.patch_dataset import InferencePatchBasedDataset
from utils import (
    inference_on_segmentation,
    compute_mean_and_std
)

# Global to set the device that handle pytorch DL computation
_DEVICE = "cpu"


def main(argv):
    with CytomineJob.from_cli(argv) as conn:
        conn.job.update(progress=0, statusComment="Initialization..")
        base_path = "{}".format(os.getenv("HOME"))  # Mandatory for Singularity
        working_path = os.path.join(base_path, str(conn.job.id))

        # Load pretrained model (assume the best of all)
        conn.job.update(progress=0,
                        statusComment="Loading segmentation model..")

        with open("/models/resnet50b_fpn256/config.json") as f:
            config = json.load(f)
        model = FPN.build_resnet_fpn(
            name=config['name'],
            input_size=conn.parameters.dataset_patch_size,  # must be / by 16
            input_channels=1 if config['input']['mode'] == 'grayscale' else 3,
            output_channels=config['fpn']['out_channels'],
            num_classes=2,  # legacy
            in_features=config['fpn']['in_features'],
            out_features=config['fpn']['out_features']
        )
        model_dict = torch.load(config['weights'],
                                map_location=torch.device(_DEVICE))
        model.load_state_dict(model_dict['model'])
        model.to(_DEVICE)

        # Select images to process
        images = ImageInstanceCollection().fetch_with_filter(
            "project", conn.parameters.cytomine_id_project)

        if conn.parameters.cytomine_id_images != 'all':
            images = [_ for _ in images if _.id
                      in map(lambda x: int(x.strip()),
                             conn.parameters.cytomine_id_images.split(','))]
        images_id = [image.id for image in images]

        # Download selected images into "working_directory"
        img_path = os.path.join(working_path, "images")
        os.makedirs(img_path)
        for image in conn.monitor(
                images, start=2, end=50, period=0.1,
                prefix="Downloading images into working directory.."):
            fname, fext = os.path.splitext(image.filename)
            if image.download(dest_pattern=os.path.join(
                    img_path, "{}{}".format(image.id, fext))) is not True:

                print("Failed to download image {}".format(image.filename))

        # create a file that lists all images (used by PatchBasedDataset
        conn.job.update(progress=50,
                        statusComment="Preparing data for execution..")
        images = os.listdir(img_path)
        images = list(map(lambda x: x+'\n', images))
        with open(os.path.join(working_path, 'images.txt'), 'w') as f:
            f.writelines(images)

        # Prepare dataset and dataloader objects
        ImgTypeBits = {'.dcm': 16}
        channel_bits = ImgTypeBits.get(fext.lower(), 8)
        mean, std = compute_mean_and_std(img_path, bits=channel_bits)

        dataset = InferencePatchBasedDataset(
            path=working_path,
            subset='images',
            patch_size=conn.parameters.dataset_patch_size,
            mode=config['input']['mode'],
            bits=channel_bits,
            mean=mean,
            std=std
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=conn.parameters.model_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            collate_fn=InferencePatchBasedDataset.collate_fn
        )

        # Go over images
        conn.job.update(status=Job.RUNNING, progress=55,
                        statusComment="Running inference on images..")
        results = inference_on_segmentation(
            model, dataloader, conn.parameters.postprocess_p_threshold)

        for id_image in conn.monitor(
                images_id, start=90, end=95,
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
                    id_terms=[conn.parameters.cytomine_id_predict_term],
                    id_project=conn.parameters.cytomine_id_project)
                )
        annotations.save()

        conn.job.update(
            status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == '__main__':
    main(sys.argv[1:])
