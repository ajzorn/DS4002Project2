# DS4002Project2


# Data

# KITTI Dataset

This repository contains information and code related to the KITTI dataset. KITTI is one of the available datasets from the dataset zoo provided by the open-source toolkit Fiftyone. The dataset can be found [here](https://www.cvlibs.net/datasets/kitti/). KITTI contains a suite of vision tasks built using an autonomous driving platform. This dataset includes the left camera images and the associated 2D object detections.

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
