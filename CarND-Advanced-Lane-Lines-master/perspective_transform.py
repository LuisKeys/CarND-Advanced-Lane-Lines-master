import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def transform_unwarp(image):
    offset = 100
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([(581, 460), (739, 460), (1070, 600), (380, 600)])
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                    [img_size[0]-offset, img_size[1]-offset], 
                                    [offset, img_size[1]-offset]])    

    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(image, M, img_size)
    return warped_img, image

def test(color_grad_img, img):
    warped_img, img = transform_unwarp(color_grad_img)
    plt.imshow(warped_img)
    plt.show()
    return warped_img, img
    