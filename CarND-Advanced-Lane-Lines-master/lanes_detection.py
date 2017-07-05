import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

# Slide window mask
def window_mask(width, height, img_ref, center,level):

    output = np.zeros_like(img_ref)
    y1 = int(img_ref.shape[0] - (level + 1) * height)
    y2 = int(img_ref.shape[0] - level * height)
    x1 = max(0,int(center-width/2))
    x2 = min(int(center+width/2), img_ref.shape[1])
    output[y1:y2, x1:x2] = 1
    return output

def find_window_centroids(warped_thres_img, window_width, window_height, margin):
    
    window_centroids = [] # Store left,right window centroid positions per level
    window = np.ones(window_width) # Create window template for convolutions
    warped = warped_thres_img[:, :, 0]
    # First find the two starting positions for the left and right lane 
    # by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    y1 = int(3 * warped.shape[0] / 4)
    x2 = int(warped.shape[1] / 2)
    l_sum = np.sum(warped[y1:, :x2], axis=0)

    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    y1 = int(3 * warped.shape[0] / 4)
    x1 = int(warped.shape[1] / 2)
    r_sum = np.sum(warped[y1:,x1:], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)
    
    # Found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    top = (int)(warped.shape[0] / window_height)
    for level in range(1,top):
        # convolve the window into the vertical slice of the image
        y1 = int(warped.shape[0] - (level + 1) * window_height)
        y2 = int(warped.shape[0] - level * window_height)
        image_layer = np.sum(warped[y1:y2,:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width / 2 as offset because convolution signal reference 
        # is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin,0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids

def detect_lanes(warped_thres_img):
    window_centroids = find_window_centroids(warped_thres_img, window_width, window_height, margin)
    warped = warped_thres_img[:, :, 0]
    detected_img = np.copy(warped_thres_img)
    img_height = warped_thres_img.shape[0]

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        centroids_count = len(window_centroids)
        ploty = np.zeros(centroids_count + 1, np.float32)
        leftx = np.zeros(centroids_count + 1, np.float32)
        rightx = np.zeros(centroids_count + 1, np.float32)

        # Go through each level and draw the windows 	
        for level in range(0, centroids_count):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

            # Fill y, left x and right x points for poly fit
            ploty[level] = img_height - level * window_height
            leftx[level] = window_centroids[level][0]
            rightx[level] = window_centroids[level][1]

	        # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        ploty[centroids_count] = 0
        leftx[centroids_count] = leftx[centroids_count - 1]
        rightx[centroids_count] = rightx[centroids_count - 1]
        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((template, zero_channel, zero_channel)), np.uint8) # make window pixels green
        detected_img = cv2.addWeighted(warped_thres_img, 1, template, 0.2, 0.0) # overlay the orignal road image with window results

        # Fit second order polynomial to centers positions
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
 
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)

    # If no window centers found, just display orginal road image
    else:
        detected_img = warped_thres_img

    return detected_img

def test(warped_thres_img, img, show_image=False):
    detected_img = detect_lanes(warped_thres_img)
    if show_image:
        plt.imshow(detected_img)
        plt.show()
    return detected_img, img
