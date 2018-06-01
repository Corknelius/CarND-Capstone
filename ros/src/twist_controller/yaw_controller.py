from math import atan


class YawController(object):
    """
    https://en.wikipedia.org/wiki/Angular_velocity

    linear_vel = angular_vel * radius (1 / curvature)
    """
    def __init__(self, wheel_base, steer_ratio,
                 min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle

    def get_angle(self, radius):
        """
        Given wheel_base and radius to achive
        calcualte the turning of wheel => the turning of steering wheel
        """
        angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_vel, angular_vel, current_vel):
        # calculate angular_vel according to current_vel
        if abs(linear_vel) > 0:
            angular_vel = current_vel * angular_vel / linear_vel
        else:
            angular_vel = 0.

        if abs(current_vel) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_vel)
            angular_vel = max(-max_yaw_rate, min(max_yaw_rate, angular_vel))

        if abs(angular_vel) > 0.:
            radius = max(current_vel, self.min_speed) / angular_vel
            return self.get_angle(radius)
        else:
            return 0.0
