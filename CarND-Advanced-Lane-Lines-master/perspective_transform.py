import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def transform_unwarp(image):
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([(304, 700), (575, 505), (770, 505), (1082, 700)])
    dst = np.float32([[350, 700], 
                      [350, 200], 
                      [1082, 200], 
                      [1082, 700]])    

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(image, M, img_size)
    return warped_img, image

def get(color_grad_img):
    warped_img, img = transform_unwarp(color_grad_img)
    return warped_img

def test(color_grad_img, img, show_image=False):
    warped_img, img = transform_unwarp(color_grad_img)
    if show_image:
        plt.imshow(warped_img)
        plt.show()
    return warped_img, img
    