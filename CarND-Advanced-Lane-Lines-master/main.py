import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cam_calibration as cc

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam()

