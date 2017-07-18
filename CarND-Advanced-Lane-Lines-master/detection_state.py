class Detection():
    def __init__(self):
        # last detection status
        self.detected = False

        # last correct left fit
        self.correct_left_xfitted = []
        # last correct right fit
        self.correct_right_xfitted = []

        # last correct left radius
        self.correct_left_xfitted = 0.0
        # last correct right radius
        self.correct_right_xfitted = 0.0

