This is a simple tutorial/memo for ROS

### ROS

More details can refer to [ROS wiki](http://wiki.ros.org/)

```bash

# run the ros master process
roscore

# running node turtle_sim_node in package turtlesim
rosrun turtlesim turtle_sim_node
rosrun turtlesim turtle_teleop_key

# list ros nodes
rosnode list

# list topic
rostopic list

# list brief info on a topic
rostopic info /turtle1/cmd_vel
Type: geometry_msgs/Twist

Publishers:
* /teleop_turtle (http://10.0.2.15:40833/)

Subscribers:
* /turtlesim (http://10.0.2.15:33825)

# show more details about type geometry_msgs/Twist
rosmsg info/show geoetry_msgs/Twist

# show documentation about the data type
rosed geometry_msgs Twist.msg

# print the topic message at real time
rostopic echo /turtle1/cmd_vel

```


### Package & Catkin workspace

#### Create a Catkin workspace

```bash

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# init catkin space
catkin_init_workspace

# check the content
ls -l

#catkin_make in root directory, i.e., catkin_ws here
catkin_make

# list again, there will be build devel src
# This setup.bash script must be sourced before using the catkin workspace.

```

Catkin workspace structure ([link](http://www.ros.org/reps/rep-0128.html))


#### Add a package

```bash

cd ~/catkin_ws/src
git clone https://github.com/udacity/simple_arm_01.git simple_arm

cd ~/catkin_ws
catkin_make

# install missing package
sudo apt-get install ros-kinetic-controller-manager

# or
source devel/setup.bash
rosdep install simple_arm

# build again
catkin_make

```

### ROS launch

```bash

source devel/setup.bash
roslaunch simple_arm robot_spawn.launch

# check dependency
rosdep check simple_arm

# install dependency
rosdep install -i simple_arm


```

### Dive into packages

```bash

# source the ros environment before creating package
cd ~/catkin_ws/src

# catkin_create_pkg <your_package_name> [dependency1 dependency2 …]

catkin_create_pkg first_package

```

I mentioned earlier that ROS packages have a conventional directory structure. Let’s take a look at a more typical package.

* scripts (python executables)
* src (C++ source files)
* msg (for custom message definitions)
* srv (for service message definitions)
* include -> headers/libraries that are needed as dependencies
* config -> configuration files
* launch -> provide a more automated way of starting nodes
* Other folders may include

* urdf (Universal Robot Description Files)
* meshes (CAD files in .dae (Collada) or .stl (STereoLithography) format)
* worlds (XML like files that are used for Gazebo simulation environments)

ROS package ([link](http://www.ros.org/browse/list.php))


### Writing ROS Nodes

#### add a ros node hello

```bash

cd ~/catkin_ws/src/simple_arm/
mkdir scripts

cd scripts
echo '#!/bin/bash' >> hello
echo 'echo Hello World' >> hello

chmod u+x hello
cd ~/catkin_ws
catkin_make
source devel/setup.bash
rosrun simple_arm hello

```

### add a service

[msg](http://wiki.ros.org/msg)
[service](http://wiki.ros.org/srv)
[CMakeList.txt](http://wiki.ros.org/catkin/CMakeLists.txt)
[package.xml](http://wiki.ros.org/catkin/package.xml)
[action/nonblocking](http://wiki.ros.org/actionlib)


The steps to add a new service

1. Define the service data format in the dir `/srv/GoToPosition.srv`
2. Change CMakeLists.txt to add the service file and its dependencies
3. Change dependencies also in package.xml
4. Write the service under `/scripts`
5. (Optinal) add parameters in launch [file](http://wiki.ros.org/roslaunch/XML), such as `robot_spawn.launch`

```bash

# check camera
rqt_image_view /rgb_camera/image_raw

# call service (tab complete)
rosservice call /arm_mover/safe_move "joint_1: 1.57
joint_2: 1.57"

```


### logging

[logging](http://wiki.ros.org/rospy/Overview/Logging)

By default all logging messages for a node are written to the node's log file
which can be found in `~/.ros/log or ROS_ROOT/log`.
If roscore is running, you can use roscd to find log
file directory by opening a new terminal window and typing:

```bash

roscd log

# see in real time in rosout
rostopic echo /rosout

rospy.init_node('my_node', log_level=rospy.DEBUG)

<!-- The look away node -->
<node name="look_away" type="look_away" pkg="simple_arm" output="screen"/>

```

ROS WIKI
ROS Answer
ROS cheetsheet
A gentle Introduction to ROS
