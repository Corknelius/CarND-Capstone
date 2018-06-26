from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color
            (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction

        #-----------------------------------
        """CAEd Notes
        Few ways to approach this:
            - Classical (what I am most familiar with)
                - Define ROIs in detected light image to isolate the three (Red/Yellow/Green) light locations
                - Convert images to specific color spaces
                    -RGB to isolate Red and Green
                    -XXX to isolate yellow (like in find lane lines assignment)
                - See which color is light (brightest 3-space value)
            - Machine Learning
                - Similar to traffic lights Assignment
                - Feed detected light image with associated label (from simulation data) to classifier
                - Save H5 file when satisfied with training
                - Load H5 during launch
        """
        #-----------------------------------
        return TrafficLight.UNKNOWN
