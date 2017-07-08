import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cam_calibration as cc
import color_gradient as cg
import perspective_transform as pt
import lanes_detection as ld
import utils as ut
import sys
import os
from moviepy.editor import VideoFileClip

mtx = ut.load_float_tensor('mtx.dat')
dist = ut.load_float_tensor('dist.dat')

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
    color_grad_img, img = cg.test(mtx, dist, 'test1.jpg', show_image=False)
    # Perspective transform
    transformed_img, img = pt.test(color_grad_img, img, show_image=False)
    # Detect lane lines
    # Determine the lane curvature
    detected_img, img = ld.test(transformed_img, img, show_image=True)
    sys.exit()

# Camera calibration
# Distortion correction
def get_cam_correction_coefs():
    ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam()

def process_image(image):
    # Color/gradient threshold
    combined, image = cg.get(image)
    # Perspective transform
    transformed_img = pt.get(image)
    # Detect lane lines
    detected_img = ld.get(transformed_img)
    return detected_img

# Callback function or video processing library
def video_pipeline_callback():
    # Main pipeline for lane lines detection
    # Color/gradient threshold
    # Perspective transform
    # Detect lane lines
    # Determine the lane curvature
    # mtx, dist = cc.test()    
    video_input = "../challenge_video.mp4"
    video_output = '../challenge_video_output.mp4'
    clip = VideoFileClip(video_input).subclip(0,2)
    # clip = VideoFileClip(videos_path + video_name)
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile(video_output, audio=False)
    sys.exit()

# test_pipeline_elements()
video_pipeline_callback()