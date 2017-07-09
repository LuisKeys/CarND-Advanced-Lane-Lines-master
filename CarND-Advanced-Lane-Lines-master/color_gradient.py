import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Color / gradient process
def abs_sobel_thresh(gray_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))

    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return grad_binary

def mag_thresh(gray_img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Apply the following steps to img
    # 1) Take the gradient in x and y separately
    abs_sobelx = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 2) Calculate the magnitude 
    abs_sobelxy = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    # 3) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255.0 * abs_sobelxy / np.max(abs_sobelxy))
    # 4) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    # 5) Return this mask as your binary_output image
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 255

    return mag_binary

def dir_thresh(gray_img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # Apply the following steps to img
    # 1) Take the gradient in x and y separately
    abs_sobelx = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 2) Take the absolute value of the x and y gradients
    abs_sobelxy = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 4) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_direction)
    # 5) Return this mask as your binary_output image
    binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1

    return binary_output

def apply_color_filter(image, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls_img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y_channel    = hls_img[:,:,0]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(y_channel)
    binary_output[(y_channel > thresh[0]) & (y_channel <= thresh[1])] = 1    
    # 3) Return a binary image of threshold result
    return binary_output

def applygradients(image, ksize):
    # Returns a 3 channel color image (all channels with the same info)
    # Apply each of the thresholding functions
    # Convert to grayscale
    cv2.rectangle(image, (0, 0), (image.shape[1], int(image.shape[0] // 1.60)), (0, 0, 0), -1)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_thresh(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    y_channel_image = apply_color_filter(image, (200, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (y_channel_image == 1)] = 255
    color_image_buffer = np.zeros((combined.shape[0],combined.shape[1],3), np.uint8)
    color_image_buffer[:, :, 0] = combined
    color_image_buffer[:, :, 1] = combined
    color_image_buffer[:, :, 2] = combined
    return color_image_buffer
    
def get(image):
    # Sobel kernel size
    ksize = 3
    # Read test image
    combined = applygradients(image, ksize)
    return combined

def test(mtx, dist, file_name, show_image=True):
    path = '../test_images/'
    # Sobel kernel size
    ksize = 3
    # Read test image
    image = mpimg.imread(path + file_name)
    combined = applygradients(image, ksize)
    if show_image:
        plt.imshow(combined)
        plt.show()
    return combined, image
