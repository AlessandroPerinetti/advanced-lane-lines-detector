# Advanced Lane Lines Detector Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project it to detect the lane lines, estimate the road curvature radius and the vehicle position.
The road that we have to face with present some non trivial complications such as curves, shadows, concrete changements, and occlusions.

![](output_video/project_video.gif)

For the achievement of the goal, the following steps were performed:

[//]: # (Image References)

[image2]: ./output_images/undistorted/test3.jpg "Road Undistorted"
[image7]: ./camera_cal/calibration1.jpg "Chessboard test"
[image8]: ./output_images/undistorted/camera_cal/calibration1.jpg "Chessboard undistorted"
[image9]: ./output_images/binary_images/X_Sobel/test3.jpg "X Sobel"
[image10]: ./output_images/binary_images/Y_Sobel/test3.jpg "Y Sobel"
[image11]: ./output_images/binary_images/Grad_Magnitude/test3.jpg "Gradient magnitude"
[image12]: ./output_images/binary_images/Grad_Direction/test3.jpg "Gradient direction"
[image13]: ./output_images/binary_images/S_Threshold/test3.jpg "S threshold"
[image13]: ./output_images/binary_images/S_Threshold/test3.jpg "S threshold"
[image17]: ./output_images/binary_images/test3.jpg "Combined Binary"
[image14]: ./output_images/warped_images/test3.jpg "Warped binary"
[image16]: ./output_images/test3.jpg "Final result"
[image18]: ./output_images/warped_images/line_pixels/test3.jpg "Line pixels"

[video1]: ./output_video/project_video.mp4 "Video"

### Camera Calibration (lines 63-91)

The first necessary operation is the camera calibration.

For this purpose, a series of chessboard images like the one below were given.

![alt text][image7]

Through the cv2.calibrateCamera() function the camera calibration and distortion coefficients are found.
Applying them to the chessboard image, the result is:

![alt text][image8]

These parameters, have to be evaluated only at the beginning of the program and then are necessary in order to undistort every frame of the video.


### Pipeline (single images)

#### 1. Undistort the image using the parameters previously found by means of the function cv2.undistort (line 587)

![alt text][image2]

#### 2. Obtain a binary image 
In order to get a meaningful binary image, different tecniques were applied.

##### 2.1 X and Y Sobel (lines 208-227)
We considered both the gradients taken the x and y the directions (min = 20, max = 100) of the grayscaled image
The results are the followings:

![alt text][image9]
![alt text][image10]

##### 2.2 Gradient magnitude and direction (lines 170-205)
Another considered parameter is the magnitude of the gradient defined as the square root of the sum of the squares of the individual x and y gradients.

![alt text][image11]

Along with the magnitude we defined also the direction of the gradient which is the arctangent of the y gradient divided by the x gradient.

![alt text][image12]

##### 2.3 HLS conversion (lines 155-168)
Since all the previous theresholding methods were computed starting from the grayscaled image, we are having loss of informations that could be useful for the line detection.
On this purpose the image is first converted into the HLS space (hue, lightness, and saturation).
After some tests, the S component appeared to be the most meaningful.
For this reason, we theresholded the S component ( min = 170, max = 255) as visibile in the image:

![alt text][image13]

##### 2.4 Combining the theresholds (lines 93- 153)
The final binary theresholded image is obtained as:
combined_binary[((xsobel_binary == 1) & (ysobel_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

![alt text][image17]

#### 3. Perform the perspective transform on the binary image. (lines 237-261)

The next thing to do is to choose four points on the image belonging to the street (src) and then define where those points have to be transformed (dst).

This source and destination choosen are the following:

| Source | Destination |
|:-------------:|:-------------:|
| 762, 500 | 980, 300 |
| 1026, 670 | 980, 420 |
| 276, 670 | 300, 420 |
| 525, 500 | 300, 300 |

By means of the function cv2.getPerspectiveTransform it is possible to evaluate a matrix M able to change the perspective of the entire image.

This transformation is done by the function cv2.warpPerspective and it is visible in the following image

![alt text][image14]

#### 4. Identification of the lane-line pixels and polynomial fitting. (lines 263-529)

Once the warped image is obtained, the next step is to identify where the lines are.
An histogram of the active pixels on the bottom half of the image is performed.
The two peaks of the histogram represents the starting point for the line search.
Starting from these points, we use the sliding windows tecnique in order to define the pixels belonging to the left and right line.

![alt text][image18]

A second order polynomial is estimated for each set of pixels.
On the next frame, it is possible to identify the new polynomials starting from a area around the previous polynomials.
If the old and new esimations differs more than 20%, we consider the new estimate not valid.
If more than 10 failing estimations in a row happens, we search the lines in all the image.


#### 5. Evaluation of the radius of curvature of the lane and the position of the vehicle with respect to center. (lines 531- 572)
From the estimated lines it is possible to evaluate the radius of curvature of the street and also the position of the car with respet to the center.


### Final result.

The final result on a test image is the following:

![alt text][image16]

### VIdeo result

The video result is visible into the folder .\\output_video


### Discussion

The pipeline has some problems in case of roads with bad conditions and has issues with the challenge videos.
