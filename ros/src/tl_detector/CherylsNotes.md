# Cheryl's Notes

## TODO:
- **Are we allowed to include the pre-trained model we wish to use for the final submission?** I recall reading that downloading the models (similar to the semantic segmentation assignment) is not allowed.
- Make sure overall project README contains team member's contact information


## Tensorflow Object Detection API

### Model Versions
Capstone Project uses TF v1.3. From Object Detection lab, the models in the provided [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) are no longer compatible with v1.3 (at the time June 2018, they are compatible with v1.5). Need to find the v1.3 models and download them locally.

#### TF v1.3 Compatible Models
Note: These models were from the [object detection lab](https://github.com/udacity/CarND-Object-Detection-Lab/blob/master/CarND-Object-Detection-Lab.ipynb)

**Went through TF Commit Logs to find model zoo version that was compatible with TF v1.4 (that worked with object detection lab)**

Model Zoo.md at [fc5145c3a8 ](https://github.com/tensorflow/models/blob/fc5145c3a8346c3b09f6268f2deccc33ef220c29/research/object_detection/g3doc/detection_model_zoo.md)

| Model name  | Speed | COCO mAP | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz) | fast | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) | fast | 24 | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)  | medium | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz) | medium | 32 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz) | slow | 37 | Boxes |
| [faster_rcnn_nas](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_17_10_2017.tar.gz) | slow | 43 | Boxes |

Inside the un-tar'ed directory, you will find:

* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference
    (try this out in the Jupyter notebook!)

## Running a Rosbag
**Use this  to run traffic light detection video from test site**

Instructions from Assignment [25.13](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/undefined/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/3251f513-2f82-4d5d-88b6-9d646bbd9101)

1. Download rviz config file
2. Open terminal and start `roscore`
3. Open new terminal and run `rosbag play -l /path/to/your.bag`
4. Open another terminal and run `rviz`
5. On rviz window: `File > Open Config > (navigate to rviz config file from step 1)`

#### Rosbag files to test against
1. [Traffic Light Detection video](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) from lesson [25.15](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/undefined/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/project)
2. [DBW test set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/reference.bag.zip) from lesson [25.6](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/undefined/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/877ed434-6955-4371-afcc-ff5b8769f0ce)

#### Extracting Rosbag Images
Motivation is for Training/Testing offline (separate of ROS, i.e. in a jupyter notebook)

## Useful Links
- Port Forwarding between VM and your simulator (on local machine) [EKF Assignment Module 3](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)
- Instructions on running Test Lot in lesson [25.6](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/reference.bag.zip)
