#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import yaml
from scipy.spatial import KDTree
import time

STATE_COUNT_THRESHOLD = 3
IS_SIM = True


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.has_image = None
        self.camera_image = None
        self.lights = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic
        light in 3D map space and helps you acquire an accurate ground truth
        data source for the traffic light classifier by sending the current
        color state of all traffic lights in the simulator. When testing on
        the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights',
                         TrafficLightArray, self.traffic_cb)
        # TODO CAEd: consider other image formats to feed into classifier
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub =\
            rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        # CAEd: set up classifier
        # full_param_name = rospy.search_param('model_location')
        # print("Full_Param Name: %s " % full_param_name)
        if IS_SIM:
            model_location = rospy.get_param(
                "/sim_model_path",
                '../../../models/ssd_sim/frozen_inference_graph.pb')
        else:
            model_location = rospy.get_param(
                "/real_model_path",
                '../../../models/ssd_real/frozen_inference_graph.pb')

        model_filter = rospy.get_param("/model_filter", 10)
        min_score = rospy.get_param("/min_score", 0.5)
        width = rospy.get_param("/width", 800)
        height = rospy.get_param("/height", 600)
        self.light_classifier =\
            TLClassifier(model_location, model_filter,
                         min_score, width, height)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.publish_traffic_light()
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d =\
                [[wp.pose.pose.position.x, wp.pose.pose.position.y]
                 for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to
            /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

    def publish_traffic_light(self):
        line_wp, state = self.process_traffic_lights()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            line_wp = line_wp if state == TrafficLight.RED else -1

            self.last_wp = line_wp
            self.upcoming_red_light_pub.publish(Int32(line_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))

        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_dx = self.waypoint_tree.query([x, y], 1)[1]

        return closest_dx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in
            styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        time0 = time.time()
        detected_state = self.light_classifier.get_classification(cv_image)
        time1 = time.time()

        print("[tl_classifer::get_classification] Time in milliseconds: ",
              (time1 - time0) * 1000)
        print("[tl_detector::get_light_state] Detected: %d, Actual: %d" % (
            detected_state, light.state))
        return detected_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a
            traffic light (-1 if none exists)
            int: ID of traffic light color (specified in
            styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None
        state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of
        # for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x,
                                                   self.pose.pose.position.y)

            # TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # stop line is the line before the traffic light
                # here we assume they have one-to-one relationship
                line = stop_line_positions[i]
                tmp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                # Find closest stop line waypoints ahead
                d = tmp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = tmp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
