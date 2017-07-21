import numpy as np

class Detection():
    def __init__(self):
        # last detection status
        self.left_detected = False
        self.right_detected = False

        # last correct left fit
        self.correct_left_xfitted = np.zeros([1, 720, 2])
        # last correct right fit
        self.correct_right_xfitted = np.zeros([1, 720, 2])

        # last correct lanes mid point
        self.bottom_lanes_mid_point = 0.0

        # last correct lanes bottom distance
        self.bottom_lanes_distance = 0.0

        # min lanes bottom distance
        self.min_bottom_lanes_distance = 20000.0

        # max lanes bottom distance
        self.max_bottom_lanes_distance = 0.0

        # last correct lanes top distance
        self.top_lanes_distance = 0.0

        # min lanes top distance
        self.min_top_lanes_distance = 20000.0

        # max lanes top distance
        self.max_top_lanes_distance = 0.0

        # last correct leftx
        self.leftx =  []

        # last correct lefty
        self.lefty =  []

        # last correct rightx
        self.rightx =  []

        # last correct righty
        self.righty =  []

        # last correct left radius
        self.left_radius =  0.0

        # last correct right radius
        self.right_radius =  0.0
