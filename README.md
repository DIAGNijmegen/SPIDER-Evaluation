# SPIDER-Evaluation
Evaluation code used for the SPIDER Challenge.

This repository provides the evaluation cope used in the SPIDER Challenge (https://spider.grand-challenge.org/). It evaluates the performance of segmentation models on the task of segmenting vertebral structures, intervertebral discs, and the spinal canal. The evaluation is based on DICE scores, a commonly used metric for assessing segmentation accuracy.

## Files and Structure
evaluation.py: This Python script calculates DICE scores for segmented structures, including vertebrae, intervertebral discs, and the spinal canal. Users can run this script to evaluate the performance of their segmentation models.

Dockerfile: Use this Dockerfile to create a Docker image for the evaluation environment. This ensures a consistent and isolated environment for running the evaluation script.

build_and_export.sh: This shell script simplifies the process of building the Docker image and exporting it for distribution. Execute this script to automate the image creation and export steps.

requirements.txt: Contains the necessary Python dependencies for running the evaluation script. You can use this file to set up a virtual environment or install the dependencies directly.

ground-truths/: This folder is a placeholder for the ground truth data. Users should add their ground truth segmentation masks to this folder when using the repository for evaluation.

## Additional Notes
Ensure that your ground truth segmentation masks have the same file names and structure as the predictions generated by your segmentation model.

Customize the evaluation script as needed to adapt it to the specific requirements of your segmentation challenge.

Feel free to modify the Dockerfile or requirements.txt to include additional dependencies required by your segmentation model.

Thank you for using the Segmentation Challenge Evaluation Repository. We hope it facilitates the evaluation process for your segmentation models. If you encounter any issues or have suggestions for improvement, please don't hesitate to reach out.
