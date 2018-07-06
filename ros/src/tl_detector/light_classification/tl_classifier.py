from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
# from utilities import label_map_util
# from utilities import visualization_utils as vis_util
# import time


class TLClassifier(object):
    def __init__(self, graph_file, class_filter, min_score,
                 TL_color_method, width, height):
        """
        TODO: Create switch for 2-step color detection method for traffic light
        defaulted to image processing based.
        need to add code for keras model (h5)
        TODO: add code to load keras model
        """

        self.min_score = min_score
        self.class_filter = class_filter
        self.TL_color_method = TL_color_method

        self.width = width
        self.height = height

        # Load Classifier

        print("[tl_classifer::init] Loading classifer located at: %s" % graph_file)
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # The following is related to the issue:
        # Crash: Could not create cuDNN handle when convnets are used
        # https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Start session as initialization takes up the most time.
        # Once loaded, detection is faster
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # The input placeholder for the image.
        # returns the Tensor with the associated name in the Graph.
        # Definite input and output Tensors for detection_graph
        self.image_tensor =\
            self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a particular object was detected.
        self.detection_boxes =\
            self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores =\
            self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes =\
            self.detection_graph.get_tensor_by_name('detection_classes:0')

        print("[tl_classifer::init] Loaded Tensorflow Graph.")

    # CAEd: The following function is from the object detection lab
    def filter_boxes(self, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= self.min_score and classes[i] == self.class_filter:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    # CAEd: The following function is from the object detection lab
    def to_image_coords(self, boxes):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * self.height
        box_coords[:, 1] = boxes[:, 1] * self.width
        box_coords[:, 2] = boxes[:, 2] * self.height
        box_coords[:, 3] = boxes[:, 3] * self.width

        return box_coords


    def detect_color(self, boxes):
        green_count = 0
        yellow_count = 0
        red_count = 0
        unknown_count = 0

        detected_color = TrafficLight.UNKNOWN

        img_boxes = self.to_image_coords(boxes)
        n = len(classes)
        for i in range(n):
            detected_color = False

            # TODO: determine method for detecting color of light given ROI
            green_count = green_count + 1
            yellow_count = yellow_count + 1
            red_count = red_count + 1

            if detected_color == false:
                unknown_count = unknown_count + 1

        if green_count > yellow_count and green_count > red_count and green_count > unknown_count:
            detected_color = TrafficLight.GREEN
        elif yellow_count > green_count and yellow_count > red_count and yellow_count > unknown_count:
            detected_color = TrafficLight.YELLOW
        elif red_count > green_count and red_count > yellow_count and green_count > unknown_count:
            detected_color = TrafficLight.RED
        else:
            detected_color = TrafficLight.UNKNOWN

        return detected_color

    def count_color(self, classes):
        green_count = 0
        yellow_count = 0
        red_count = 0
        unknown_count = 0

        detected_color = TrafficLight.UNKNOWN

        n = len(classes)
        for i in range(n):
            class_detected = classes[i]
            if class_detected == 1:
                green_count = green_count + 1
            elif class_detected == 2:
                red_count = red_count + 1
            elif class_detected == 3:
                yellow_count = yellow_count + 1
            else:
                unknown_count = unknown_count + 1

        if green_count > yellow_count and green_count > red_count and green_count > unknown_count:
            detected_color = TrafficLight.GREEN
            print( "[tl_classifer::count_color] Green")
        elif yellow_count > green_count and yellow_count > red_count and yellow_count > unknown_count:
            detected_color = TrafficLight.YELLOW
            print( "[tl_classifer::count_color] Yellow")
        elif red_count > green_count and red_count > yellow_count and green_count > unknown_count:
            detected_color = TrafficLight.RED
            print( "[tl_classifer::count_color] Red")
        else:
            detected_color = TrafficLight.UNKNOWN
            print( "[tl_classifer::count_color] Unknown")

        return detected_color

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color
            (specified in styx_msgs/TrafficLight)

        """
        """CAEd Notes
        Few ways to approach this:
            - Classical (what most familiar with) (2-step)detected_light_state
                - Define ROIs in detected light image to isolate the three
                  (Red/Yellow/Green) light locations
                - Convert images to specific color spaces
                    -RGB to isolate Red and Green
                    -XXX to isolate yellow (like in find l
            if detected_color == false:ane lines assignment)
                - See which color is lit (brightest 3-space value)
            - Machine Learning (2-step)
                - Similar to traffic signs Assignment
                - Feed detected light image with associated label
                      (from simulation data) to classifier
                - Save H5 file when satisfied with training
                - Load H5 during launch
            - Deep learning (1-step)
                - Have one classifier that detects and classifies light
        """

        detected_light_state = TrafficLight.UNKNOWN

        # Actual detection.
        with self.detection_graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [self.detection_boxes,
                 self.detection_scores, self.detection_classes],
                feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if len(boxes) > 0:
            if self.class_filter > -1:
                # implement 2-step traffic light detection
                filtered_boxes, filtered_scores, filtered_classes =\
                    self.filter_boxes(boxes, scores, classes)
                detected_light_state = self.detect_color(boxes)
            elif self.TL_color_method == 1:
                # ignore most-confident model class and manually detect color
                detected_light_state = self.detect_color(boxes)
            elif self.TL_color_method == 2:
                # model was pretrained to determine color of light as class,
                # count most detected value
                detected_light_state = self.count_color(classes)
            else:
                # model was pretrained to determine color of light as class,
                # get most confident value
                most_confident_detection = classes[0]
                if most_confident_detection == 1:
                    detected_light_state = TrafficLight.GREEN
                elif most_confident_detection == 2:
                    detected_light_state = TrafficLight.RED
                elif most_confident_detection == 3:
                    detected_light_state = TrafficLight.YELLOW
                else:
                    detected_light_state = TrafficLight.UNKNOWN

        return detected_light_state
