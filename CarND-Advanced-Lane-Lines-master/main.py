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
this = sys.modules[__name__]
this.frames_counter = 0

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
    color_grad_img, img = cg.test(mtx, dist, 'output_ori_10.png', show_image=False)
    # Perspective transform
    transformed_img= pt.test(color_grad_img, img, show_image=False)
    # Detect lane lines
    # Determine the lane curvature
    detected_img = ld.test(transformed_img, img, show_image=True)
    sys.exit()
 
# Camera calibration
# Distortion correction
def get_cam_correction_coefs():
    ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam()

def process_image(image):

    this.frames_counter += 1
    # Color/gradient threshold
    combined = cg.get(image)
    # Perspective transform
    transformed_img = pt.get(combined)
    # Detect lane lines
    detected_img, warped_img = ld.get(transformed_img, image)
    if this.frames_counter % 10 == 0:
        cv2.imwrite("../output_images/output_ori_" + str(frames_counter) + ".png", image)
        cv2.imwrite("../output_images/output_comb_" + str(frames_counter) + ".png", combined)
        cv2.imwrite("../output_images/output_transf_" + str(frames_counter) + ".png", transformed_img)
        cv2.imwrite("../output_images/output_det_" + str(frames_counter) + ".png", detected_img)
        cv2.imwrite("../output_images/output_warp_" + str(frames_counter) + ".png", warped_img)

    return warped_img

# Callback function or video processing library
def video_pipeline():
    # Main pipeline for lane lines detection
    # Color/gradient threshold
    # Perspective transform
    # Detect lane lines
    # Determine the lane curvature
    #ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam()
    video_input = "../challenge_video.mp4"
    video_output = '../challenge_video_output.mp4'
    clip = VideoFileClip(video_input).subclip(0, 2)
    #clip = VideoFileClip(video_input)
    this.frames_counter = 0
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile(video_output, audio=False)
    sys.exit()

#test_pipeline_elements()
video_pipeline()