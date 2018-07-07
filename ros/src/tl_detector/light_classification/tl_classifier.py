from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np


class TLClassifier(object):
    """
    TODO: Create switch for 2-step color detection method for traffic light
        defaulted to image processing based.
        need to add code for keras model (h5)
    TODO: add code to load keras model

    https://github.com/udacity/CarND-Object-Detection-Lab/blob/master/
    CarND-Object-Detection-Lab.ipynb
    """
    def __init__(self, graph_file,
                 class_filter, min_score, width, height):
        self.min_score = min_score
        self.class_filter = class_filter

        self.width = width
        self.height = height

        # Load Classifier
        print("[tl_classifer::init] Loading classifer: %s" % graph_file)
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # The input placeholder for the image.
            # returns the Tensor with the associated name in the Graph.
            # Definite input and output Tensors for detection_graph
            self.image_tensor =\
                self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes =\
                self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores =\
                self.detection_graph.get_tensor_by_name('detection_scores:0')
            # The classification of the object (integer id).
            self.detection_classes =\
                self.detection_graph.get_tensor_by_name('detection_classes:0')

        # The following is related to the issue:
        # Crash: Could not create cuDNN handle when convnets are used
        # https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Start session as initialization takes up the most time.
        # Once loaded, detection is faster
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        print("[tl_classifer::init] Loaded Tensorflow Graph.")

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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color
            (specified in styx_msgs/TrafficLight)
        """
        with self.detection_graph.as_default():
            img_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes) = self.sess.run(
                [self.detection_boxes,
                 self.detection_scores, self.detection_classes],
                feed_dict={self.image_tensor: img_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if len(boxes) > 0:
            most_confident_detection = classes[0]
            if most_confident_detection == 1:
                print("GREEN")
                return TrafficLight.GREEN
            elif most_confident_detection == 2:
                print("RED")
                return TrafficLight.RED
            elif most_confident_detection == 3:
                print("YELLOW")
                return TrafficLight.YELLOW

            print("UNKNOWN")
            return TrafficLight.UNKNOWN
