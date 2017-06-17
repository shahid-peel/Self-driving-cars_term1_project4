## Writeup

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

[image0a]: ./my_examples/distorted_test_image.jpg "Distorted Test Image"
[image0b]: ./my_examples/undistorted_test_image.jpg "Undistorted Test Image"
[image1]: ./my_examples/distorted_image.jpg "Distorted"
[image2]: ./my_examples/undistorted_image.jpg "Undistorted"
[image22]: ./test_images/test1.jpg "Road Transformed"
[image3a]: ./my_examples/color_binary.jpg "Color Binary"
[image3b]: ./my_examples/combined_binary.jpg "Combined Binary"
[image4]: ./my_examples/warped.jpg "Warped Image"
[image5]: ./my_examples/poly_fit.jpg "Poly Fit Visual"
[image6]: ./my_examples/result.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./P4.ipynb".  The functions that performs parts of the calibration are:
- get_image_and_object_points()
- calibrate_camera()

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to a test image using the `cv2.undistort()` function.

Before:
![alt text][image0a]

After:
![alt text][image0b]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of the distorted (original) image:
![alt text][image1] 

And below is the undistorted image.
![alt text][image2]

You can see that the correction results in some parts of the image going out of the original boundary. Its like stretching the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In function `s_color_and_gradient_threshold()`, which appears in the 1st code cell of this IPython notebook, I use thresholding on the S channel (of HLS) and sobel filter in the horizontal direction (x) to create a binary image. Here's an example of my output:

Binary Image:
![alt text][image3a]   

Combined binary Image:
![alt text][image3b]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `create_warped_image_and_display()`, which appears in the 1st code cell of this IPython notebook. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[705, 450],[1200, 720],[190, 720], [590,450]])
x = 250
y = 0
width = 780
height = 720
dst = np.float32([[x+width,y],[x+width,y+height],[x,y+height],[x,y]])
```

My source and destination points look like this:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 250, 0        | 
| 190, 720      | 250, 720      |
| 1200, 720     | 1030, 720      |
| 705, 450      | 1030, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using the function `sliding_window_fit_polynomial()`, which appears in the 2nd code cell of this IPython notebook, I performed a sliding window search for lane pixels and later used it to fit a 2nd order polynomial. This polynomial gives me the curve of the lane line.  Since this is applied on the perspective corrected image, i.e. birds eye view, the curve reflects the actual curve on the ground and is not effected by the original perspective projection (where parallel lines may not remain parallel).  

Here is the output of the polynomial fitting.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This also happend in the `sliding_window_fit_polynomial()` function, which appears in the 2nd code cell of this IPython notebook. The code for calculating the curvature is below:

```python
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    img_bottom = warped.shape[0]-1
    left_lane_x = left_fit[0]*img_bottom**2 + left_fit[1]*img_bottom + left_fit[2]
    right_lane_x = right_fit[0]*img_bottom**2 + right_fit[1]*img_bottom + right_fit[2]
    car_center_offset = round(((right_lane_x + left_lane_x)/2 - warped.shape[1]/2)*xm_per_pix, 2)
    
    # Fit new polynomials to x,y in world space
    left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    #y_eval = warped.shape[0]-1 #np.max(ploty)
    y_eval = np.max(ploty)/2
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Now our radius of curvature is in meters
    #print(round(left_curverad,0), 'm', round(right_curverad,0), 'm')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Curvature (in meters) = {}'.format(int((left_curverad+right_curverad)/2)),(10,50), font, 1.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Vehicle is {}m {} of center'.format(abs(car_center_offset), 'left' if car_center_offset < 0 else 'right'),(10,110), font, 1.5,(255,255,255),2,cv2.LINE_AA)
```

Radius of curvature is calculated by the equation:

curve = ((1+(2Ay+B)^2)^3/2)/∣2A∣

The position of the vehicle is calculated by looking at the bottom most part of the left and right curves. Their x coordinates converted from pixels to meters gives the position w.r.t center of car.  These values are then superimposed on top of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this also in the `sliding_window_fit_polynomial()` function, which appears in the 2nd code cell of this IPython notebook. 

Code for result plotting back onto road is below:

```python
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minverse, (binary_warped.shape[1], binary_warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
```

Here is an example of my result on a test image:
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./my_examples/result_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Had some issues with the RGB vs. BGR format the different apis use for images read (e.g. video processing was using different format) but it was solved easily.

Hypothetical cases where pipeline can fail:
- Cars coming in the lane ahead of our car
- Camera gets uncalibrated, either moves or the vibration cause the lens to move, which would give us either distorted image or wrong values of curvature
- If the road is not close to flat, e.g. curving/climbing up sharply, the perspective corrected images will not give parallel lines
- Rain causing contrast of line and road to decrease
- dust on camera lens
- Construction work areas where lane lines may not always be parallel
- If the car ever moves out of a lane so that the projection corrected image does not show to lane lines, then the solution would break down.

What could be improved:
- Calibration on images that are closer, as cv2 couldn't find corner points in all the calibration images.  Also more images for calibration might be better. On the same point, we can measure the parallel vertical and horizontal lines for a test image that was taken head on (orthogonal to the camera ray).
- We assume that the camera doesn't have a tilt in the assembly with car. There should be a verification or some kind, or extra calibration that can give us some error, however small.
- Using more filters for lane detection (beyond sobel and s channel).

Also the solution will need to be tested in night conditions as that is as important as daylight testing.  