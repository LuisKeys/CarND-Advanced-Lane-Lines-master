**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration logic was implemented in the cam_calibration.py module of the project.
In main.py (line 42):
	ret, mtx, dist, rvecs, tvecs = cc.calibrate_cam() 
can be called to get the mtx and dist coeficients.
Lines 25 to 27 (currently commented) get those values and store them in mtx.dat and dist.dat files
to have them handy without the need to process them each time.

calibrate_cam() function (line 27 of cam_calibration.py) basically does the following:
1) Define nx = 9 and ny = 6
2) Walkthrouh all the calibration images
3) Call calibrate_cam_by_img(image_file, nx, ny, obj_points, img_points, obj_p, index) (line 9 of cam_calibration.py) on each iteration,
wich:
	a) Read Image
	b) Convert BGR to Gray
	c) Call ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None) to get board corners 
	d) If corners were found then call:
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
	e) And finally undistort: 
		dst = cv2.undistort(img, mtx, dist, None, mtx)

A test() function is also provided on this module (and all the others) to test individual images to tune the whole system
Check sample images in the next section.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
#### Distorted image:
![image1]( ./camera_cal/calibration2.jpg "Distorted image")

After the process described in the previous section (Camera Calibration)
#### Corrected image:
![image2]( ./camera_cal/output_calibration2.png "Corrected image")

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

All the process for color and gradients is implemented in color_gradient.py module.
The main function is applygradients(img, ksize) (line 98 of this module), and does the following:
1) Add black masks for areas not required for this recognition (as everything on top of the horizon line and at the bottom 
   of the image where the car is displayed)
2) Creates a mask for colos near black (very dark gray range) to exclude those points (dark_mask_binary)
3) Converts BGR image in Gray scale
4) Get horiz and vert gradients (gradx and grady) 
   with the abs_sobel_thresh(gray_img, orient='x', sobel_kernel=3, thresh=(0, 255)) function (line 7)
   with a kernel size of 31 (top limit value)
5) values are normalized to 0 or 1 (binary image) based on a threshold range (30, 100), using cv2.Sobel function
6) Grad magnitude is calculated with mag_thresh(gray_img, sobel_kernel=3, mag_thresh=(0, 255)) based on a threshold range (30, 100), 
   function (line 30), using cv2.Sobel function
7) Calculate de gradient direction with dir_thresh(gray_img, sobel_kernel=3, thresh=(0, np.pi/2)) function, with (0.7, 1.3) range threshold
8) Apply a color filter converting BGR image to HLS color space and using a 200 as lower threshold for lightness channel and 
   66 for saturation channel 
9) Finally all the previous values are combined as follows:
	
	combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (channel_image == 1)) & (dark_mask_binary == 1)] = 255

The result is an image as the following:

#### Binary color and gradient filtered image:
![image2]( ./output_images/output_comb_460.png "Binary color and gradient filtered image")

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
