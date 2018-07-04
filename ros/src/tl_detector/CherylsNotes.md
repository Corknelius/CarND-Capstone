# Cheryl's Notes

## TODO:
- Make sure overall project README contains team member's contact information
- Are we allowed to include the pre-trained model we wish to use for the final submission?
  - I recall reading that downloading the models (similar to the semantic segmentation assignment) is not allowed.
- Need to see if referenced pre-trained models work with TF v1.3 or figure out a way to port it to work with TF v1.3

----------------------------
Run the Project
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

--------
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

#### Extracting Rosbag Images & save to video
Motivation is for Training/Testing offline (separate of ROS, i.e. in a jupyter notebook)

See this useful post from [GitHub](https://stackoverflow.com/questions/22346013/how-to-extract-image-frames-from-a-bagfile)


1. In a directory in which is writable (i.e. ~/catkin_ws/bagfiles) type in the terminal
```
rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_raw
```
2. Then play the bagfile.(Though it can be done before or after) Then in the terminal in which rosrun was executed the following appears:
```
> [ INFO] [1394806321.162974947]: Saved image frame0467.jpg
" The frames were made in that directory. process completed."
```
3. Save to video
```
ffmpeg -framerate 25 -pattern_type glob -i "*.jpg" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p carla_tl_loop.mp4
```

#### Saving rostopic to text files
```
rosbag play mybag.bag
rostopic echo /foo > output.txt
```
------------------------
## labelImg
This is a tool to annotate imagery for the classifier. Tool may be found here:
https://github.com/tzutalin/labelImg

First you will need to check out the git repository


Then you will need to go to that repository directory


Then use the following command to create or run a docker container with the labelImg program
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
----------------------------
## Bosch Datasets
Note that the bosch data set is in parts. You will need to put the partial zip files in a common directory and fuse them together.

Link to datasets:
https://hci.iwr.uni-heidelberg.de/node/6132/download/4230201a07fab2fb9acdcee4f2dd9cd8

**NOTE**: This link will be accessible until Wed, 07/04/2018 - 07:54. If you need
access after the link expires, don't hesitate to revisit the download page on
https://hci.iwr.uni-heidelberg.de/

**Training Data**
```
cd /home/cheryl/Development/TL_data/bosch/train
cat dataset_train_rgb.zip.* > dataset_train_rgb.zip
```
**Testing Data**
```
cd /home/cheryl/Development/TL_data/bosch/test
cat dataset_test_rgb.zip.* > dataset_test_rgb.zip
```

**Strategy: MANGLE YAML Label DATA!**
You will need to go through the .yaml files to find and replace labels
```
RedLeft -> Red
RedRight -> Red  
RedStraight -> Red
GreenLeft -> Green
GreenRight -> Green
GreenStraight -> Green
```

**convert dataset to TF Record file**
Bosch Training Set
```
python create_tf_record.py \
 --data_dir=/home/cheryl/Development/TL_model_make/data/bosch/train/train.yaml \
  --output_path=/home/cheryl/Development/TL_model_make/data/bosch/bosch_train.record \
   --label_map_path=/home/cheryl/Development/TL_model_make/data/udacity_label_map.pbtxt
```
Bosch Testing Set
```
python create_tf_record.py \
 --data_dir=/home/cheryl/Development/TL_model_make/data/bosch/test/test.yaml \
  --output_path=/home/cheryl/Development/TL_model_make/data/bosch/bosch_test.record \
   --label_map_path=/home/cheryl/Development/TL_model_make/data/udacity_label_map.pbtxt
```
-----
--------------
ProtoBuf 3.4
https://github.com/google/protobuf/releases/tag/v3.4.0
