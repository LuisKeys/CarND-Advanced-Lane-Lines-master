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
import detection_state 
from moviepy.editor import VideoFileClip

mtx = ut.load_float_tensor('mtx.dat')
dist = ut.load_float_tensor('dist.dat')
this = sys.modules[__name__]
this.frames_counter = 0
this.detection = detection_state.Detection()

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

# Callback function or video processing library
def process_image(image):

    # distortion correction
    image = cv2.undistort(image, mtx, dist, None, mtx)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    this.frames_counter += 1
    # Color/gradient threshold
    combined = cg.get(image_bgr)
    # Perspective transform
    transformed_img = pt.get(combined)

    # Detect lane lines
    detected_img, warped_img = ld.get(transformed_img, image, this.detection)

    xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
    car_offset = 1280.0 / 2.0 - this.detection.bottom_lanes_mid_point + 60 # offset in pixles
    car_offset *= xm_per_pix # offset in meters

    if this.frames_counter % 30 == 0:

        cv2.imwrite("../output_images/output_ori_" + str(frames_counter) + ".png", image_bgr)
        cv2.imwrite("../output_images/output_comb_" + str(frames_counter) + ".png", combined)
        cv2.imwrite("../output_images/output_transf_" + str(frames_counter) + ".png", transformed_img)
        cv2.imwrite("../output_images/output_det_" + str(frames_counter) + ".png", detected_img)
        warped_img_output = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)

        #Add radius
        cv2.putText(warped_img_output,"Left radius=" + "{:4.0f}".format(this.detection.left_radius) + " m", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(warped_img_output,"Right radius=" + "{:4.0f}".format(this.detection.right_radius) + " m", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        #Add car offset
        cv2.putText(warped_img_output,"Car offset=" + "{:3.1f}".format(car_offset) + " m", (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.imwrite("../output_images/output_warp_" + str(frames_counter) + ".png", warped_img_output)

        #print('Bottom dist:' + str(this.detection.bottom_lanes_distance))
        #print('Min Bottom dist:' + str(this.detection.min_bottom_lanes_distance))
        #print('Max Bottom dist:' + str(this.detection.max_bottom_lanes_distance))

        #print('Top dist:' + str(this.detection.top_lanes_distance))
        #print('Min Top dist:' + str(this.detection.min_top_lanes_distance))
        #print('Max Top dist:' + str(this.detection.max_top_lanes_distance))

    #Add radius
    cv2.putText(warped_img,"Left radius=" + "{:4.0f}".format(this.detection.left_radius) + " m", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    cv2.putText(warped_img,"Right radius=" + "{:4.0f}".format(this.detection.right_radius) + " m", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

    #Add car offset
    cv2.putText(warped_img,"Car offset=" + "{:3.1f}".format(car_offset) + " m", (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

    return warped_img

# Video process
def video_pipeline():
    # Main pipeline for lane lines detection
    # Color/gradient threshold
    # Perspective transform
    # Detect lane lines
    # Determine the lane curvature
    #ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam()
    video_input = "../project_video.mp4"
    video_output = '../project_video_final_output.mp4'
    #clip = VideoFileClip(video_input).subclip(48, 50)
    clip = VideoFileClip(video_input)
    this.frames_counter = 0
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile(video_output, audio=False)

    sys.exit()

#test_pipeline_elements()
video_pipeline()