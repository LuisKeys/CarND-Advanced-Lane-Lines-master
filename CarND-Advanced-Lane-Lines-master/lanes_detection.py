import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def detect_lanes(warped_thres_img):

    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical sections
    margin = 100 # Slide left and right for searching

    # Use histogram first
    warped = np.copy(warped_thres_img[:, :, 0])
    warped_thres_img_copy = np.copy(warped_thres_img)
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    top = (int)(warped.shape[0] / window_height)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    # Set height of windows
    window_height = np.int(warped.shape[0] / top)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    # Set minimum number of pixels found to recenter window
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Iterate windows
    for window in range(top):
        # Get window boundaries
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw windows
        cv2.rectangle(warped_thres_img_copy, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
        cv2.rectangle(warped_thres_img_copy, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 
        # Get nonzero pixels within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
       # If you found, recenter next window
        if len(good_left_inds) > window_width:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > window_width:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # x and y values to plot
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0] )
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    warped_thres_img_copy[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    warped_thres_img_copy[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    window_img = np.zeros_like(warped_thres_img_copy)
    # Margin points
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))    

    # Center points
    left_center_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_center_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

    # Draw center lines in window img
    cv2.polylines(window_img,np.int_([left_center_line]), False, (255, 255, 0), 4)
    cv2.polylines(window_img,np.int_([right_center_line]), False, (255, 255, 0), 4)

    # Mix window image with warped image (that already has the boxes added)
    detected_img = cv2.addWeighted(warped_thres_img_copy, 1, window_img, 1, 0)

    return detected_img

def test(warped_thres_img, img, show_image=False):
    detected_img = detect_lanes(warped_thres_img)
    if show_image:
        plt.imshow(detected_img)
        plt.show()
    return detected_img, img
