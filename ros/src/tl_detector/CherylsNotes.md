# Cheryl's Notes

## TODO:
- Make sure overall project README contains team member's contact information
- Are we allowed to include the pre-trained model we wish to use for the final submission?
  - I recall reading that downloading the models (similar to the semantic segmentation assignment) is not allowed.
- Need to see if referenced pre-trained models work with TF v1.3 or figure out a way to port it to work with TF v1.3

----------------------------

## Tensorflow Object Detection API

#### Model Versions
Capstone Project uses TF v1.3. From Object Detection lab, the models in the provided [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) are no longer compatible with v1.3 (at the time June 2018, they are compatible with v1.5). Need to find the v1.3 models and download them locally.

#### TF v1.3 Compatible Models
Note: These models were referenced in the [object detection lab](https://github.com/udacity/CarND-Object-Detection-Lab/blob/master/CarND-Object-Detection-Lab.ipynb)

I went through TF Commit Logs to find model zoo version that was compatible with TF v1.4 (that worked with object detection lab)

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
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.inde/x`, `model.ckpt.meta`)
* a frozen graph proto with weights baked into the graph as constants
  (`frozen_inference_graph.pb`) to be used for out of the box inference
    (try this out in the Jupyter notebook!)

----------------------------

## Running a Rosbag
**Use this  to run traffic light detection video from test site**

Instructions from Assignment [25.13](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/3251f513-2f82-4d5d-88b6-9d646bbd9101)

1. Download [rviz config file](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/default.rviz)
2. Open terminal and start `roscore`
3. Open new terminal and run `rosbag play -l /path/to/your.bag`
4. Open another terminal and run `rviz`
5. On rviz window: `File > Open Config > (navigate to rviz config file from step 1)`

#### Rosbag files to test against
1. [Traffic Light Detection video](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) from lesson [25.15](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/undefined/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/project)
2. [DBW test set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/reference.bag.zip) from lesson [25.6](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/undefined/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/877ed434-6955-4371-afcc-ff5b8769f0ce)

#### Extracting Rosbag Images
Motivation is for Training/Testing offline (separate of ROS, i.e. in a jupyter notebook)

See this useful post from [GitHub](https://stackoverflow.com/questions/22346013/how-to-extract-image-frames-from-a-bagfile)


This was my image topic /front_camera/camera/image_raw/compressed. This is what i had to do:

1. In a directory in which is writable (~/catkin_ws/bagfiles) type in the terminal
```
rosrun image_view extract_images image:=/front_camera/camera/image_raw _image_transport:=compressed
```
2. Then play the bagfile.(Though it can be done before or after) Then in the terminal in which rosrun was executed the following appears:
```
> [ INFO] [1394806321.162974947]: Saved image frame0467.jpg
" The frames were made in that directory. process completed."
```

#### Saving rostopic to text files
```
rosbag play mybag.bag
rostopic echo /foo > output.txt
```
----------------------------

## Useful Links
#### Port Forwarding between VM and your simulator (on local machine)
[EKF Assignment Module 3](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

#### Instructions on running Test Lot in lesson [25.6](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/877ed434-6955-4371-afcc-ff5b8769f0ce)

The CarND-Capstone/ros/src/waypoint_loader/launch/waypoint_loader.launch file is set up to load the waypoints for the first track. To test using the second track, you will need to change

```xml
<param name="path" value="$(find styx)../../../data/wp_yaw_const.csv" />
```

to use the churchlot_with_cars.csv as follows:
```xml
<param name="path" value="$(find styx)../../../data/churchlot_with_cars.csv"/>
```
Note that the second track does not send any camera data.


---------------------------------
## rostopic

#### Topics when running Simulator
```
/base_waypoints
/current_pose
/current_velocity
/final_waypoints
/image_color
/rosout
/rosout_agg
/tf
/tf_static
/traffic_waypoint
/twist_cmd
/vehicle/brake_cmd
/vehicle/brake_report
/vehicle/dbw_enabled
/vehicle/lidar
/vehicle/obstacle
/vehicle/obstacle_points
/vehicle/steering_cmd
/vehicle/steering_report
/vehicle/throttle_cmd
/vehicle/throttle_report
/vehicle/traffic_lights
```

#### Topics in DBW bagfile

#### Topics in Traffic Light bagfile

------------------------
## labelImg
This is a tool to annotate imagery for the classifier. Tool may be found here:
https://github.com/tzutalin/labelImg


Use the following command to create or run a docker container with the labelImg program
```
sudo docker run -it \
--user $(id -u) \
-e DISPLAY=unix$DISPLAY \
--workdir=$(pwd) \
--volume="/home/$USER:/home/$USER" \
--volume="/etc/group:/etc/group:ro" \
--volume="/etc/passwd:/etc/passwd:ro" \
--volume="/etc/shadow:/etc/shadow:ro" \
--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
tzutalin/py2qt4
```

Then use the following command to create the program (you can skip this if you already done so)
```
make qt4py2
```

The call the program
```
./labelImg.py
```
