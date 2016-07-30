# Visual Relationship Detection with Language Priors

## Introduction
Visaul-Relationship-Detection is system that detects visual relationship (including subject, predicate, object) given a RGB image.

Detailed description of the system will be provided by our technical report at ECCV 2016 website http://www.ranjaykrishna.com/index.html

"Visual Relationship Detection with Language Priors",
Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
(ECCV 2016), 2016(oral). (* = indicates equal contribution)

##Licence

This software is being made available for research purpose only. Check LICENSE file for details.

## System Requirements

This software is tested on Ubuntu 14.04 LTS (64bit).

Prerequisites

MATLAB (tested with 2014b on 64-bit Linux)
prerequisites for caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)

## Demo
You can directly run demo.m about 50 images with single relationship detection result will be shown. Resulting images will be saved in "results/demo" and "results/demo_zeroShot".
```
>> run demo.m;
```
## Relationship and Preidcate Detection 
To detect relationship in our relationship dataset (http://www.ranjaykrishna.com/index.html), please run relationship_phrase_detection.m. That is, each image will have a set of relationship trupe <subject, predicate, object > with localtion (subject and object bounding boxes)  The relationship and phrase detection result (including zero-shot preformance) also will be reported. Preidacate detection result is by running predicate_detection.m. We provide our VVG based object detection results (category, confident score and bound boxes) and   CNN score on union of the boundingboxes of the two participating objects in that relationship. You also can train your object detection model with other archetechure such as ResNet. 
```
>> run relationship_phrase_detection.m;
>> run predicate_detection.m;
```

 



