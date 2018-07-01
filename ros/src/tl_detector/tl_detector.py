#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import time

STATE_COUNT_THRESHOLD = 3


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
        #TODO CAEd: you may want to consider other image formats to feed into classifier
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub =\
            rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        #-----------------------------
        # CAEd: set up classifier
        #------------------------------
        model_location = rospy.get_param('~model_location', '../../../models/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb')
        model_filter = rospy.get_param('~model_filter', 10)
        min_score = rospy.get_param('~min_score', 0.5)
        TL_color_method = rospy.get_param('~TL_color_method', 1)
        TL_color_model = rospy.get_param('~TL_color_model', 'None')
        roi_x = rospy.get_param('~roi_x', 0)
        roi_y = rospy.get_param('~roi_y', 0)
        roi_width = rospy.get_param('~roi_width', 800 )
        roi_height = rospy.get_param('~roi_height', 600)
        self.light_classifier = TLClassifier(model_location,model_filter, min_score,
                            TL_color_method, TL_color_model,
                            roi_x, roi_y, roi_width, roi_height)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # TODO: will need to process traffic lights
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
        #TODO CAEd: update code to include slow-down when seeing yellow light
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            #if state == TrafficLight.RED: then stop
            #elif state == TrafficLight.YELLOW: then slow down
            # QUESTION: Should there be more logic for lining up with the line WP?
            #elif state == TrafficLight.GREEN: then go
            #else: ???
            self.last_state = self.state
            '''
            if light status is red or yellow, set the waypoint to line_wp
            otherwise, -1
            '''
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
        # TODO implement
        # CAEd walkthrough recommends reuse code from section 1 KD Trees
        # CAEd: Not sure what exactly needs to be implemented. KD tree is
        # being used and function is properly called.
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


        # TODO: Is there a parameter for prelim testing vs using simulator vs using real data?
        # For testing, just return the light state
        # rospy.logwarn("light state {}".format(light.state))


        #TODO CAEd: implement section below when not testing.

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        time0 = time.time()
        detected_state =  self.light_classifier.get_classification(cv_image)
        time1 = time.time()

        print("[tl_classifer::get_classification] Time in milliseconds: ", (time1 - time0) * 1000)
        print("[tl_detector::get_light_state] Detected: %d, Actual: %d" % (detected_state, light.state))
        
        # CAEd: ONLY FOR TESTING
        detected_state = light.state

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
