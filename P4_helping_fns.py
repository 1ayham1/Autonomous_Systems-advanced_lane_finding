import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


"""****************************************************************************"""
def cal_undistort(img, objpoints, imgpoints):
    """
        1. takes an image, object points, and image points
        2. performs the camera calibration, image distortion correction
        3. returns the undistorted image
        
    """
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Camera calibration, given object points, image points, and the shape of the grayscale image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    #Undistorting a test image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist


"""****************************************************************************"""
def abs_sobel_thresh(one_channel_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
        Calculate directional gradient for a one-channel image
    """
    
    # 1) Take the derivative in x or y
    if orient == 'x':
        soble_ = cv2.Sobel(one_channel_img, cv2.CV_64F, 1, 0)
    if orient == 'y':
        soble_ = cv2.Sobel(one_channel_img, cv2.CV_64F, 0, 1)
    
    # 2) take the absolute value
    abs_sobel = np.absolute(soble_)
    
    # 3) Scale to 8-bit (0 - 255)
    scale_factor = np.max(abs_sobel)/255 
    scaled_sobel = (abs_sobel/scale_factor).astype(np.uint8) 
    
    # 4) Create a binary mask of ones where threshold is met 
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
 
    return grad_binary

"""****************************************************************************"""

def mag_thresh(one_channel_img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
        Calculate gradient magnitude of a one-channel image
    """
    
    # 1) Take the derivative in x or y
    sobelx = cv2.Sobel(one_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(one_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 2) Calculate the gradient magnitude
    mag_soble = np.sqrt(sobelx**2 + sobely**2)
    
    # 3) Rescale to 8 bit (0-255) and convert to type=np.uint8
    scale_factor = np.max(mag_soble)/255.0 
    mag_soble = (mag_soble/scale_factor).astype(np.uint8) 
        
    # 4) Create a binary mask of ones where threshold is met 
    mag_binary = np.zeros_like(mag_soble)
    mag_binary[(mag_soble >= mag_thresh[0]) & (mag_soble <= mag_thresh[1])] = 1

    # 5) Return the binary image
    
    return mag_binary
                                            
"""****************************************************************************"""

def dir_threshold(one_channel_img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    """
        Calculate gradient direction of a one channel image
    """ 
    
    
    # 1) Take the derivative in x or y
    sobelx = cv2.Sobel(one_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(one_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # 2) take the absolute value
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely) 
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
                                            
    # 3) Create a binary mask of ones where threshold is met 
    dir_binary =  np.zeros_like(grad_direction)
    dir_binary[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
                                      
    return dir_binary
"""****************************************************************************"""

def binary_Channel_Select(channel, thresh=(0, 255)):
    '''
        output a binary image after applying a threshold to a one-channel image.
        Used to visually inspect certain behaviours
    '''
    
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output

"""****************************************************************************"""

def show_images(image_list, titles, plot_size, size = (2,2) ):
    
    img_width, img_hight = image_list[0].shape
    rows,cols = size[0], size[1]
    
    images = image_list.reshape(rows,cols,img_width,img_hight)
    lables = titles.reshape(rows, cols)
    
      
    f1, ax = plt.subplots(rows, cols ,figsize=plot_size)
    f1.tight_layout()
    f1.subplots_adjust(hspace=0.15)

    for i in range(rows):
        for j in range(cols):
            
            counter = i*cols +j
            ax[i][j].imshow(images[i][j],'gray')
            ax[i][j].set_title(str(counter)+": " + lables[i][j], color ='r', fontweight='bold')
            ax[i][j].axis('off')

            
"""****************************************************************************"""

def edges_pipe(one_ch_img, thr, ksize = 3):
    
    L_th = thr[0]
    H_th = thr[1]
    
    gradx = abs_sobel_thresh(one_ch_img, orient='x', sobel_kernel=ksize, thresh=(L_th, H_th))
    grady = abs_sobel_thresh(one_ch_img, orient='y', sobel_kernel=ksize, thresh=(L_th, H_th))
    mag_binary = mag_thresh(one_ch_img, sobel_kernel=ksize, mag_thresh=(L_th, H_th))
    dir_binary = dir_threshold(one_ch_img, sobel_kernel=ksize, thresh=(0.2, 1.1)) #(0,np.pi/2)
    
    return np.array([gradx,grady,mag_binary, dir_binary ])

"""****************************************************************************"""
def get_Threshold(imgs_of_concern):
    #configuring various parameters
    #------------------------------------------
    #threshold list: calculated emperically after visually inspecting various alternatives
    #Utilizes Otsu's robust method for determining the dual threshold value

    thr_vec = []
    for c_image in imgs_of_concern: 
        th, bw = cv2.threshold(c_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thr_vec.append(th)
        
    parameters = {}
   
    #name of channlel: low_threshold, high_threshold, kernel_size
    parameters['gray']= [np.round(0.5*thr_vec[0]) ,np.round(1.5*thr_vec[0]) ,3 ]
    parameters['S_hls']= [np.round(0.5*thr_vec[1]) ,np.round(1.5*thr_vec[1]) , 5 ]
    parameters['L_lab']= [np.round(0.5*thr_vec[2]), np.round(1.5*thr_vec[2]) , 7 ]
    parameters['B_lab']= [np.round(0.5*thr_vec[3]), np.round(1.5*thr_vec[3]) , 5 ]
    parameters['L_luv']= [np.round(0.5*thr_vec[4]), np.round(1.5*thr_vec[4]) , 5 ]
    parameters['R_rgb']= [np.round(0.5*thr_vec[4]), np.round(1.5*thr_vec[4]) , 5 ]
    

    return parameters
    
    
"""****************************************************************************"""
# The following function Pipeline the above steps and is used later in vedio processing
# The following function Pipeline the above steps and is used later in vedio processing
def get_binaryImageFeatures(image):

    #Extract Features from color images after applying a threshold
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5)) # clipLimit=2.0, tileGridSize=(8,8)
    equ_img = clahe.apply(gray)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    H_hsv = hsv[:,:,0]
    S_hsv = hsv[:,:,1]
    V_hsv = hsv[:,:,2]

    #Examining Various color spaces

    L_hls = hls[:,:,1]
    S_hls = hls[:,:,2] 

    L_lab = lab[:,:,0]
    B_lab = lab[:,:,2] 


    L_luv = luv[:,:,0]

    R_rgb = image[:,:,0]
    G_rgb = image[:,:,1]

    gray_binary = binary_Channel_Select(gray, thresh=(187, 255)) #187
    S_hls_binary = binary_Channel_Select(S_hls, thresh=(191, 255)) #191
    R_rgb_binary = binary_Channel_Select(R_rgb, thresh=(219, 255)) # 219
    B_lab_binary = binary_Channel_Select(B_lab, thresh=(165, 255))  # [146-179] only left lane
    L_lab_binary = binary_Channel_Select(L_lab, thresh=(207, 255)) #207
    L_luv_binary = binary_Channel_Select(L_luv, thresh=(213, 255)) # 213

    #additional
    S_hsv_binary = binary_Channel_Select(S_hsv, thresh=(77, 250)) # for yellow
    V_hsv_binary = binary_Channel_Select(V_hsv, thresh=(230, 250))  # for white

    detect_yellow =  (B_lab_binary == 1)| (S_hls_binary == 1)&(S_hsv_binary==1) 
    detect_white = ((L_lab_binary == 1) & (L_luv_binary == 1))| (V_hsv_binary==1)
    detect_both =  ((gray_binary == 1) & (R_rgb_binary == 1) )
    combined_thresh = np.zeros_like(gray_binary)

    #combining images with best features by visually inspecting the previous figures
    combined_thresh[(detect_yellow | detect_white | detect_both )] =1

   
    kernel = np.ones((3,3),np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    
    return combined_thresh




def get_ColorEdgeFeatures(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(5,5)) # clipLimit=2.0, tileGridSize=(8,8)
    equ_img = clahe.apply(gray)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    #Examining Various color spaces
    S_hls = hls[:,:,2] 
    L_lab = lab[:,:,0]
    B_lab = lab[:,:,2] 
    L_luv = luv[:,:,0]
    R_rgb = image[:,:,0]

    
    gray_binary = binary_Channel_Select(gray, thresh=(187, 255)) 
    S_hls_binary = binary_Channel_Select(S_hls, thresh=(191, 255)) 
    R_rgb_binary = binary_Channel_Select(R_rgb, thresh=(219, 255)) 
    B_lab_binary = binary_Channel_Select(B_lab, thresh=(165, 255)) 
    L_lab_binary = binary_Channel_Select(L_lab, thresh=(207, 255)) 
    L_luv_binary = binary_Channel_Select(L_luv, thresh=(213, 255)) 
    
    
    combined_thresh = np.zeros_like(gray_binary)

    #combining images with best features by visually inspecting the previous figures
    detect_yellow = ((gray_binary == 1) | (S_hls_binary == 1) & (B_lab_binary == 1) )
    detect_white = ((R_rgb_binary == 1) | (L_lab_binary == 1) & (L_luv_binary == 1))

    combined_thresh[(detect_yellow & detect_white )] =1

    kernel = np.ones((3,3),np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
    #---------------------------------------------------------------------------
    #Sobel Contribution
    
    channels_of_concern = [gray, S_hls, L_lab, B_lab, L_luv,R_rgb ]
    channels_names = ['gray', 'S_hls', 'L_lab','B_lab', 'L_luv', 'R_rgb']
    labels = np.array(['_GradX','_GradY' , '_ThMag','_GradDir' ])

    all_images = []
    all_labels = []
    parameters ={}

    parameters = get_Threshold(channels_of_concern)

    
    for i in range(len(channels_names)):

        img_ch_name = channels_names[i]

        L_th = parameters[img_ch_name][0]
        H_th = parameters[img_ch_name][1]
        ksize = parameters[img_ch_name][2]

        grad_images = edges_pipe(channels_of_concern[i], [L_th, H_th], ksize)
        all_images.extend(grad_images)

        nLables = [w.replace('_','th=(' + str(L_th) + ',' + str(H_th) + ')| ' +
                             channels_names[i] + '_' ) for w in labels]
        all_labels.extend(nLables)

        combine_soble = np.zeros_like(gray_binary)

    #combining images with best features by visually inspecting the previous figures that shows sobel edges and gradients

    x_grad_combine = (((all_images[4] == 1) & (all_images[20] == 1))| (all_images[12] == 1) )
    y_grad_combine = ((all_images[5] == 1) & (all_images[13] == 1))
    th_mag_combine = (((all_images[2] == 1) & (all_images[18] == 1) & (all_images[22] == 1))  | (all_images[14] == 1) )
    grad_dir_combine = ((all_images[11] == 1) & (all_images[19] == 1)| (all_images[15] == 1) )

    combine_soble[   
                (th_mag_combine & grad_dir_combine ) | (y_grad_combine & x_grad_combine)
            ] =1

    kernel = np.ones((3,3),np.uint8)
    combined = cv2.morphologyEx(combine_soble, cv2.MORPH_CLOSE, kernel)

    #----------------------------------------------------------------------------
    
    combined = np.zeros_like(gray_binary)

    combined[   
                (combine_soble==1) | (combined_thresh ==1) 
        ] =1

    kernel = np.ones((3,3),np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    #----------------------------------------------------------------------------
    
    
    return combined

"""****************************************************************************"""
#Define perspective transform function
def warp(img, src, dst):

    img_size = (img.shape[1], img.shape[0])
   
    M = cv2.getPerspectiveTransform(src, dst) # perspective matrix
    #Minv = cv2.getPerspectiveTransform(dst, src) #inverse perspective transform
    
    #create wraped image -uses linear interpolation
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped 

"""****************************************************************************"""
def draw_Lines(img,T_Left,B_Left, B_Right, T_Right,color= [255, 0, 0], thickness=10):
    img_Lines = np.copy(img)
    
    cv2.line(img_Lines,(int(B_Right[0]),int(B_Right[1])), (int(T_Right[0]),int(T_Right[1])),color, thickness)
    cv2.line(img_Lines,(int(B_Left[0]),int(B_Left[1])), (int(T_Left[0]),int(T_Left[1])),color, thickness)
    cv2.line(img_Lines,(int(B_Right[0]),int(B_Right[1])), (int(B_Left[0]),int(B_Left[1])),color, thickness)
    cv2.line(img_Lines,(int(T_Left[0]),int(T_Left[1])), (int(T_Right[0]),int(T_Right[1])),color, thickness)
    
    return img_Lines


"""****************************************************************************"""
def identify_LaneLines(binary_warped):
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
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
        # Draw the windows on the visualization image

        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
   

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    
    return out_img,leftx,lefty,rightx,righty, left_fit, right_fit


"""****************************************************************************"""
def draw_Full_DetectedPath(image, binary_warped,left_fit,right_fit, src,dst ):
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src) #inverse perspective transform

    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return result

"""****************************************************************************"""

