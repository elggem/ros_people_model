# ros_peoplemodel

XXX Insert Youtube Video XXX

This is a perception architecture meant for social artificial intelligence and robotics. It combines various Deep Learning classifiers to build a model of which people are present and their attributes. Working so far are modules for:

  - Detection of 68 Face landmarks to be used by classifiers based on Dlib 68 Landmark
  - Face ID using the 128D vector embedding from Dlib, in addition with some simple clustering logic
  - Emotion recognition based on [LINK]
  - Eyes closed detection based on [LINK]

![Diagram Architecture](https://raw.githubusercontent.com/elggem/ros_slopp/master/images/arch.png)

## Roadmap
  - Integration with a classifier for speaking detection
  - Integration with ...
  - Integration with OpenPose and various classifiers for
    - Body pose estimation (sitting, standing, waving, etc.)
    - Hand pose estimation (open palm, fist, etc.)
  - Integration wih Masked RCNN architectures to detect various categories of objects in the hands of people.


## Setup


## Dependencies
### Dlib
Cuda!
### ORB2_SLAM
Instructions...



## Usage

Describe here!
