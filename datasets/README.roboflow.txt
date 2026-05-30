
deteksi barang - v2 tes
==============================

This dataset was exported via roboflow.com on May 30, 2026 at 3:51 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2569 images.
Deteksi-barang are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none
* Random brigthness adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 3 pixels


