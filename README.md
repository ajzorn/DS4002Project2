# DS4002Project2

**Hypothesis:** After analyzing results from both models, YOLOv8 will be our recommended model for/better identifying cars, time of day, weather, and pedestrians in different photographed images of Charlottesville. This hypothesis is based on YOLOv8's higher accuracy on the COCO dataset, its anchor-free model that speeds up detection in crowded scenes, and its specialized training augmentations, which could improve its performance in an urban environment like Charlottesville. Additionally, YOLOv8's quicker inference time and efficient resource utilization suggest its suitability for real-time object detection in this specific context. However, practical testing is necessary to validate this hypothesis conclusively. .

**Research Question:** How can we use image entity recognition to identify cars, time of day, weather, and pedestrians in different photographed images in order to aid Charlottesville traffic enforcement and patterns?

**Model Approach:** We will attempt to use two object detection models, one called Detectron2 and then TorchVision’s Faster R-CNN (YOLOV8) model. We will train both models on the same training sets, in which entities such as cars and pedestrians are identified in addition to weather and time of day. The model will be measured using metrics like a confusion matrix, F1 score, precision, recall.

**Executive Summary:** We will train two object detection models YOLOV8 and Detectron 2 on images containing traffic patterns, which includes vehicles and pedestrians.  Using these two object detection models, we will try to find the better performing model for the question about identifying Charlottesville traffic patterns with team-chosen images. 



# Data

# KITTI Dataset

This repository contains information and code related to the KITTI dataset. KITTI is one of the available datasets from the dataset zoo provided by the open-source toolkit Fiftyone. The dataset can be found [here](https://www.cvlibs.net/datasets/kitti/). KITTI contains a suite of vision tasks built using an autonomous driving platform. This dataset includes the left camera images and the associated 2D object detections.

**Data Dictionary**:
Media type:  image
Num samples: 7481
Persistent:  False
Tags:        []
Sample fields:
    id:           fiftyone.core.fields.ObjectIdField
    filepath:     fiftyone.core.fields.StringField
    tags:         fiftyone.core.fields.ListField(fiftyone.core.fields.StringField)
    metadata:     fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.metadata.ImageMetadata)
    ground_truth: fiftyone.core.fields.EmbeddedDocumentField(fiftyone.core.labels.Detections)


## Dataset Details

- Training Split: 7,481 annotated images
- Test Split: 7,518 unlabeled images

We will also collect additional data ourselves from Google Images.

## Data Source

The KITTI dataset can be accessed [here](https://www.cvlibs.net/datasets/kitti/).

## Google Images Data

The images in the `cville_roads` folder were collected from a Google Images search.

## Example Code

Below is an example code snippet that demonstrates how to load the KITTI dataset using Fiftyone:

```python
#!pip install fiftyone
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("kitti", split="train")
