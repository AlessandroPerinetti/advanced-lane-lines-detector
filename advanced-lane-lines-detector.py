#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import pandas as pd
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML

'''
===============================================================================
CLASS DEFINITION
===============================================================================
'''
# Define a class to receive the characteristics of each line detection
class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # xe  values of the last n fits of the line
        self.recent_xfitted = []    
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]   
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
'''
===============================================================================
FUNCTION DEFINITION
===============================================================================
'''

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    
    return cv2.addWeighted(initial_img, α, img, β, γ)

def grayscale(img):
    """Return an image in grayscale
       To show the image: plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def find_chessboard_corners(img,objpoints,imgpoints,objp):       
    # Convert the image to grayscale
    gray = grayscale(img)
    # Find chessboard  corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If corners are found, add the object points and image points to the array
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        #Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
    return

def get_distortion_param():
    global image_folder
    '''
    Obtain the distortion parameters of the camera with the chessboard calibration 
    '''
    #Read the list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    objpoints = [] # points in real world space 
    imgpoints = [] # points in image space

    # Define the object points (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    for fname in images:
        # Read each image in the calibration folder
        img = mpimg.imread(fname)     
        find_chessboard_corners(img, objpoints, imgpoints, objp)    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    
    undistorted_subfolder = 'undistorted/'
    for fname in images:
        img = mpimg.imread(fname) 
        undistorted_img = cv2.undistort(img,mtx,dist,None, mtx)
        plt.imsave(image_folder + undistorted_subfolder + fname,undistorted_img)
    
    return  mtx, dist

def get_binary_image(img):   
    '''
    Function that, given an undistorted image, returns a binary theresholded image 
    using different methods
    '''
    global process_choice
    global image_folder
    
    s_binary = HLS_threshold(img) 
    mag_binary = mag_threshold(img)
    dir_binary = dir_threshold(img)
    xsobel_binary = abs_sobel_thresh(img, orient = 'x')
    ysobel_binary = abs_sobel_thresh(img, orient = 'y')

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(mag_binary)
    combined_binary[((xsobel_binary == 1) & (ysobel_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
    
#    #Show the different influences on the combined binary image
#    combined_binary_1 = np.zeros_like(mag_binary)
#    combined_binary_1[(xsobel_binary == 1) & (ysobel_binary == 1)] = 1
#    combined_binary_2 = np.zeros_like(mag_binary)
#    combined_binary_2[(mag_binary == 1) & (dir_binary == 1)] = 1 
#    f, axes = plt.subplots(3, 3, figsize=(32, 16))
#    axes[0,0].set_title('Gradient magnitude', fontsize=30)
#    axes[0,0].imshow(mag_binary)
#    axes[0,1].set_title('Gradient direction', fontsize=30)
#    axes[0,1].imshow(dir_binary)
#    axes[0,2].set_title('Gradient mag + dir', fontsize=30)
#    axes[0,2].imshow(combined_binary_2)
#    axes[1,0].set_title('X Sobel', fontsize=30)
#    axes[1,0].imshow(xsobel_binary)
#    axes[1,1].set_title('Y Sobel', fontsize=30)
#    axes[1,1].imshow(ysobel_binary)
#    axes[1,2].set_title('X + Y Sobel', fontsize=30)
#    axes[1,2].imshow(combined_binary_1)
#    axes[2,0].set_title('Original', fontsize=30)
#    axes[2,0].imshow(img)
#    axes[2,1].set_title('S component', fontsize=30)
#    axes[2,1].imshow(s_binary)
#    axes[2,2].set_title('Combined', fontsize=30)
#    axes[2,2].imshow(combined_binary)
    
    # Save the images in the defined folders
    if process_choice == 'i':
        global image_name
        global image_count
        
        binary_img_subfolder = 'binary_images/'
        X_Sobel_subfolder = 'X_Sobel/'
        Y_Sobel_subfolder = 'Y_Sobel/'
        Grad_Magnitude_subfolder = 'Grad_Magnitude/'
        Grad_Direction_subfolder = 'Grad_Direction/'
        S_subfolder = 'S_Threshold/'
        plt.imsave(image_folder + binary_img_subfolder + X_Sobel_subfolder + image_name[image_count],xsobel_binary, cmap='gray')
        plt.imsave(image_folder + binary_img_subfolder + Y_Sobel_subfolder + image_name[image_count],ysobel_binary, cmap='gray')
        plt.imsave(image_folder + binary_img_subfolder + Grad_Magnitude_subfolder + image_name[image_count],mag_binary, cmap='gray')
        plt.imsave(image_folder + binary_img_subfolder + Grad_Direction_subfolder + image_name[image_count],dir_binary, cmap='gray')
        plt.imsave(image_folder + binary_img_subfolder + S_subfolder + image_name[image_count],s_binary, cmap='gray')

    return combined_binary

def HLS_threshold(img):
    ''' Convert to HLS color space and separate the S channel
     Note: img is the undistorted image
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    return s_binary

def mag_threshold(img, sobel_kernel=9, mag_thresh=(30, 100)):
    '''
    Function that returns the binary image computed using the gradient magnitude medthod
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

def dir_threshold(img, sobel_kernel= 15, thresh=(0.7, 1.3)):
    '''
    Function that returns the binary image computed using the gradient direction medthod
    '''
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def abs_sobel_thresh(img, orient, thresh_min=20, thresh_max=100):
    ''' 
    Define a function that takes an image, and perform its Sobel operation
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Apply the threshold to the binary output
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def get_perspective_matrix(img,src,dst):
    '''
    Given an image, the source and destination points, obtain the perspective matrix
    '''
    M = cv2.getPerspectiveTransform(src,dst)
    
    return M

def change_perspective(binary_img):
    #Choose the four points that define the road plane for the perspective change
    global src
    global dst
    global img_size
    
    #Plot the position of the points 
    #plt.figure(figsize=(20,10))
    #plt.imshow(binary_img)
    #plt.plot(276,670,'.', markersize=30)
    #plt.plot(1026,670,'.', markersize=30)
    #plt.plot(525,500,'.', markersize=30)
    #plt.plot(762,500,'.', markersize=30)
    # ATTENTION: the shape vector is inverted
    
    img_size = (binary_img.shape[1],binary_img.shape[0]) 
        
    src = np.float32([[762,500],[1026,670],[276,670],[525,500]])
    dst = np.float32([ [img_size[0]-300, 300], [img_size[0]-300, img_size[1]-100],
                      [300, img_size[1]-100],[300, 300]])
        
    M = get_perspective_matrix(binary_img,src,dst)
    warped_img = cv2.warpPerspective(binary_img,M,img_size,flags=cv2.INTER_LINEAR)
    
    return warped_img
    
def find_lane_pixels(binary_warped):
    '''
    Function that, given a binary warped image, defines the possible 
    pixels belonging to the left and right line
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 100

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
#        # Draw the windows on the visualization image
#        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
#        (win_xleft_high,win_y_high),(0,255,0), 2) 
#        cv2.rectangle(out_img,(win_xright_low,win_y_low),
#        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    '''
    Function that, starting from the histogram peaks of the warped image, 
    estimate the polynomial coefficients.
    '''
    global n
    global process_choice
    global image_folder
    global xm_per_pix  
    global ym_per_pix

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    right_line.current_fit = np.polyfit(righty, rightx, 2)
    
    left_line.recent_xfitted.insert(0,left_line.current_fit)
    right_line.recent_xfitted.insert(0,right_line.current_fit)
    
    if len(left_line.recent_xfitted) > n:
        left_line.recent_xfitted.pop()
    if len(right_line.recent_xfitted) > n:
        right_line.recent_xfitted.pop()
        
    left_line.best_fit = np.array([0,0,0], dtype='float') 
    for left_count in range(0, len(left_line.recent_xfitted)):
        left_line.best_fit = left_line.best_fit + left_line.recent_xfitted[left_count]
        
    left_line.best_fit = left_line.best_fit/len(left_line.recent_xfitted)
    
    right_line.best_fit = np.array([0,0,0], dtype='float')
    for right_count in range(0, len(right_line.recent_xfitted)):
        right_line.best_fit = right_line.best_fit + right_line.recent_xfitted[right_count] 
        
    right_line.best_fit = right_line.best_fit/len(right_line.recent_xfitted)   
        
    left_line.detected = True
    right_line.detected = True

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
        right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_line.detected = False
        right_line.detected = False
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Save the images in the defined folders
    if process_choice == 'i':
        global image_name
        global image_count
        
        warped_img_subfolder = 'warped_images/'
        line_pixels_subfolder = 'line_pixels/'
        plt.imsave(image_folder + warped_img_subfolder + line_pixels_subfolder + image_name[image_count],out_img)

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    result = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return result, left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    
    global n
    global n_left
    global n_right
    global max_missed_detection
    global xm_per_pix  
    global ym_per_pix

     #Fit a second order polynomial to each with np.polyfit() ###
    left_line.current_fit = np.polyfit(lefty, leftx, 2)
    right_line.current_fit = np.polyfit(righty, rightx, 2)
    
    if all(left_line.current_fit > 1.2*left_line.recent_xfitted[0]) or all(left_line.current_fit < 0.8*left_line.recent_xfitted[0]):
        left_line.current_fit = left_line.recent_xfitted[0]
        n_left = n_left + 1
        if n_left >= max_missed_detection:
            left_line.__init__()

        
    if all(right_line.current_fit > 1.2*right_line.recent_xfitted[0]) or all(right_line.current_fit < 0.8*right_line.recent_xfitted[0]):
        right_line.current_fit = right_line.recent_xfitted[0]
        n_right = n_right + 1 
        if n_right >= max_missed_detection:
            right_line.__init__()
        
    left_line.recent_xfitted.insert(0,left_line.current_fit)
    right_line.recent_xfitted.insert(0,right_line.current_fit)

    
    if len(left_line.recent_xfitted) > n:
        left_line.recent_xfitted.pop()
    if len(right_line.recent_xfitted) > n:
        right_line.recent_xfitted.pop()

    left_line.best_fit = np.array([0,0,0], dtype='float') 
    for left_count in range(0, len(left_line.recent_xfitted)):
        left_line.best_fit = left_line.best_fit + left_line.recent_xfitted[left_count]
        
    left_line.best_fit = left_line.best_fit/len(left_line.recent_xfitted)
    
    right_line.best_fit = np.array([0,0,0], dtype='float')
    for right_count in range(0, len(right_line.recent_xfitted)):
        right_line.best_fit = right_line.best_fit + right_line.recent_xfitted[right_count] 
        
    right_line.best_fit = right_line.best_fit/len(right_line.recent_xfitted)   
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
    right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr

def search_around_poly(binary_warped):
    # Choose the width of the margin around the previous polynomial to search
    margin = 50

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #Set the area of search based on activated x-values within the +/- margin 
    #of the polynomial function found on the previous frame
    left_lane_inds = ((nonzerox > (left_line.best_fit[0]*(nonzeroy**2) + left_line.best_fit[1]*nonzeroy + 
                    left_line.best_fit[2] - margin)) & (nonzerox < (left_line.best_fit[0]*(nonzeroy**2) + 
                    left_line.best_fit[1]*nonzeroy + left_line.best_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_line.best_fit[0]*(nonzeroy**2) + right_line.best_fit[1]*nonzeroy + 
                    right_line.best_fit[2] - margin)) & (nonzerox < (right_line.best_fit[0]*(nonzeroy**2) + 
                    right_line.best_fit[1]*nonzeroy + right_line.best_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    result = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return result, left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    global ym_per_pix  # meters per pixel in y dimension
    global xm_per_pix # meters per pixel in x dimension
        
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def measure_car_position(ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    global ym_per_pix  
    global xm_per_pix
    global img_size
    global left_x_base
    global right_x_base
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    left_x_base = left_line.best_fit[0]*y_eval**2 + left_line.best_fit[1]*y_eval + left_line.best_fit[2]
    right_x_base = right_line.best_fit[0]*y_eval**2 + right_line.best_fit[1]*y_eval + right_line.best_fit[2]
    
    base_distance_pixel = right_x_base - left_x_base
    centre_position_pixel  = img_size[0]/2
    
    car_position = (base_distance_pixel - centre_position_pixel)*xm_per_pix
    
    return car_position

def process_image(img):
    '''
        Define the pipeline 
    '''
    global distortion_param
    global process_choice
    global image_folder
    global raw_image
    global image_name
    global image_count
    global car_position

    # Undistort the image using the distortion parameters found with the camera calibration
    undistorted_img = cv2.undistort(img,distortion_param[0],distortion_param[1],None, distortion_param[0])

    # Obtain the binary image of the original frame
    combined_binary = get_binary_image(undistorted_img)

    # Obtain the bird's eye view of the binary image
    warped_img = change_perspective(combined_binary)
    
    # Define a polynomial estimation of the lane lines
    if left_line.detected == True and right_line.detected == True :
        out = search_around_poly(warped_img) 
    else:
        out = fit_polynomial(warped_img)
        
    # Find radius of curvature and car position
    left_line.radius_of_curvature , right_line.radius_of_curvature = measure_curvature_real(out[3], out[4], out[5])
    left_line.line_base_pos = measure_car_position(out[3])
    cv2.putText(undistorted_img,'Radius of curvature: {:.0f}'.format(left_line.radius_of_curvature), (300,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)
    cv2.putText(undistorted_img,'Vehicle is {:.2f} m left of center'.format(left_line.line_base_pos), (300,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)

    # Return to the original perspective
    Minv = get_perspective_matrix(undistorted_img,dst,src)
    unwarped_img = cv2.warpPerspective(out[0],Minv,img_size,flags=cv2.INTER_LINEAR)
    
    #Overlap the original image with the detected lines
    result = cv2.addWeighted(undistorted_img, 1, unwarped_img, 0.3, 0)    
        
    # Save the images in the defined folders
    if process_choice == 'i':
        binary_img_subfolder = 'binary_images/'
        warped_img_subfolder = 'warped_images/'
        undistorted_img_subfolder = 'undistorted/'
        plt.imsave(image_folder + binary_img_subfolder + image_name[image_count],combined_binary, cmap='gray')
        plt.imsave(image_folder + warped_img_subfolder + image_name[image_count],warped_img, cmap='gray')
        plt.imsave(image_folder + undistorted_img_subfolder + image_name[image_count],img)

    return result

'''
===============================================================================
BEGIN
===============================================================================
'''

# Definition of the left and right line as Line classes
left_line = Line()
right_line = Line()
car_position = 0
left_x_base = 0
left_x_base = 0

#Variables definition 
image_folder = 'output_images/'
src = np.float32([])
dst = np.float32([])
img_size = []
n = 20
max_missed_detection = 10
n_left = 0
n_left = 0
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Obtain the distortion parameters of the camera
distortion_param = get_distortion_param()

#Decide if process the images or the videos
print('Choose if process images or videos (i/v)')
process_choice = input()

if process_choice == 'i':
    '''
    Process the images
    '''
    raw_image = []
    image_name = []
    
    # Read the test images
    raw_image.append(mpimg.imread('test_images/straight_lines1.jpg'))
    image_name.append('straight_lines1.jpg')
    
    raw_image.append(mpimg.imread('test_images/straight_lines2.jpg'))
    image_name.append('straight_lines2.jpg')
    
    raw_image.append(mpimg.imread('test_images/test1.jpg'))
    image_name.append('test1.jpg')
    
    raw_image.append(mpimg.imread('test_images/test2.jpg'))
    image_name.append('test2.jpg')
    
    raw_image.append(mpimg.imread('test_images/test3.jpg'))
    image_name.append('test3.jpg')
    
    raw_image.append(mpimg.imread('test_images/test4.jpg'))
    image_name.append('test4.jpg')
    
    raw_image.append(mpimg.imread('test_images/test5.jpg'))
    image_name.append('test5.jpg')
    
    raw_image.append(mpimg.imread('test_images/test6.jpg'))
    image_name.append('test6.jpg')
    
    for image_count in range(0, len(raw_image)):
        result = process_image(raw_image[image_count])
        plt.imsave(image_folder + image_name[image_count],result)
        left_line.__init__()
        right_line.__init__()
    
if process_choice == 'v':
    '''
    Process the videos
    '''
    output_1 = 'output_video/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    #clip1 = VideoFileClip("project_video.mp4").subclip(5,15)
    simple_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    simple_clip.write_videofile(output_1, audio=False)
