import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def transform_unwarp(image, inv=False, mask=True):
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([(377, 648), (526, 512), (808, 514), (1079, 650)])
    dst = np.float32([[350, 700], 
                      [350, 400], 
                      [1082, 400], 
                      [1082, 700]])    

    M = cv2.getPerspectiveTransform(src, dst)
    if inv:
        M = cv2.getPerspectiveTransform(dst, src)

    warped_img = cv2.warpPerspective(image, M, img_size)

    if mask:
        # top
        cv2.rectangle(warped_img, (0, 0), (120, image.shape[0]), (0, 0, 0), -1)
        # bottom
        cv2.rectangle(warped_img, (1150, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)

    return warped_img

def get(color_grad_img):
    warped_img = transform_unwarp(color_grad_img, False, True)
    return warped_img

def get_warp(img):
    warped_img = transform_unwarp(img, True, False)
    return warped_img

def test(color_grad_img, img, show_image=False):
    warped_img= transform_unwarp(color_grad_img)
    if show_image:
        plt.imshow(warped_img)
        plt.show()
    return warped_img
    