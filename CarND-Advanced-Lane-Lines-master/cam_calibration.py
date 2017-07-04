import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_cam_by_img(image_file, nx, ny, obj_points, img_points, obj_p, index):
    # Read in an image
    img = cv2.imread(image_file)
    # Convert to gray color space
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get chessboard points 
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    print(ret)
    if ret:
        # If chessboard was recognized, do calibration process
        img_points.append(corners)
        obj_points.append(obj_p)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        print("Calibration image: " + str(index) + " processed")
        return gray.shape[::-1]
   

def calibrate_cam():
    # Camera calibration
    nx = 9
    ny = 6
    path = '../camera_cal/'
    file_name = 'calibration*.jpg'
    # Load all calibration images
    images_files = glob.glob(path + file_name)
    obj_points = []
    img_points = []
    obj_p = np.zeros((ny * nx, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # Iterate all images
    print("Calibration images count:" + str(len(images_files)))
    index = 0
    img_shape = []
    for image_file in images_files:
        img_shape = calibrate_cam_by_img(image_file, nx, ny, obj_points, img_points, obj_p, index)
        index += 1

    # Get camera coeficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

    # Test cam calibration
    test_img = cv2.imread(path + 'calibration2.jpg')
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_undist = cv2.undistort(gray, mtx, dist, None, mtx)
    plt.imshow(test_undist, cmap="gray")
    plt.show()
    return ret, mtx, dist, rvecs, tvecs