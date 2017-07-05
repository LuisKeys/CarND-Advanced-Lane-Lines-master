import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cam_calibration as cc
import color_gradient as cg
import perspective_transform as pt
import lanes_detection as ld
import utils as ut

def test_pipeline_elements():
    # Main pipeline for lane lines detection
    # Camera calibration
    # Distortion correction
    # mtx, dist = cc.test()    
    # ut.save_float_tensor('mtx.dat', mtx)
    # ut.save_float_tensor('dist.dat', dist)
    # Color/gradient threshold
    mtx = ut.load_float_tensor('mtx.dat')
    dist = ut.load_float_tensor('dist.dat')
    color_grad_img, img = cg.test(mtx, dist, 'straight_lines1.jpg', show_image=False)
    # Perspective transform
    transformed_img, img = pt.test(color_grad_img, img, show_image=False)
    # Detect lane lines
    detected_img, img = ld.test(transformed_img, img, show_image=True)
    # Determine the lane curvature

# Camera calibration
# Distortion correction
def get_cam_correction_coefs():
    ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam()

# Callback function or video processing library
def video_pipeline_callback():
    # Main pipeline for lane lines detection
    # Color/gradient threshold
    # Perspective transform
    # Detect lane lines
    # Determine the lane curvature
    pass

test_pipeline_elements()
