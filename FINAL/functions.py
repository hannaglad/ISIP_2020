import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import (measure, transform, feature, filters, color, draw)
import cv2
import imutils
import math
from scipy.spatial.distance import pdist, squareform
from skimage.util import invert
from scipy.optimize import fmin
import warnings

def rmse(vec1, vec2):
    """Takes in 2 vectors and calculates the root mean square error"""
    diff = vec1-vec2
    return np.sqrt(np.dot(diff, diff)/len(vec1))

def display(image):
    """Function to quickly display a grascale image with matplotlib"""
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()
    return 0

def full_preprocess(img):
    """Performes normalization, bilateral filtering, gaussian blur and otsu thresholding on a grayscale image"""

    # Normalization
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    # Bilater filtering
    img = cv2.bilateralFilter(img, 11, 700, 700)
    # Gaussian blurring
    img = cv2.GaussianBlur(img, (11,11),4)
    # Otsu thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def binar_otsu(image):
    """Performs otsu binarization on a grayscale image"""
    ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2


def searchspaceMin(pre_binarized, post, circles):
    """Function to set region of image that does not contain electrodes to a mean of the image color
    Inputs : binarized image of pre-operative cochlea, post operative image of implant, and center the circle of implants
    Outputs : Image with the outside of the circle set to the mean image color
    """

    pre_binarized = 255-pre_binarized
    # Get x;y coordinates and radius
    x,y,r = circles[0,:][0]

    # Create an image of the same shape as the original
    rows, cols = post.shape
    mask = np.ones((post.shape[0], post.shape[1]))
    # Calculate the mean value of the image
    mean_value = np.mean(post)

    # If the pixel is inside the radius of the Hough circle;
    # Set the mask value to the same value as the original image
    # Otherwise, set it to the mean image value
    for i in range(cols):
        for j in range(rows):
            if math.hypot(i-x, j-y) > r:
                mask[j,i] = mean_value
            else :
                mask[j,i] = post[j,i]

    return mask

def fit_spiral(center_pre,x,y):
    """ Function to fit a logaritmic spiral through data points at the pre-defined center
    Inputs : Center (X-Y coordinates), Data point coordinates X, Data point coordinates Y
    Outputs : Values for fitting a spiral in exponential, polar coordinate form
    """
    # fit a log spiral at the specified center
    y = y - center_pre[1]
    x = x - center_pre[0]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    theta = np.unwrap(theta) # correction for angle mapping
    solution = np.polyfit(theta,np.log(r),1)    # spiral : log(r) = log(a) + b*theta
    afit = np.exp(solution[1])
    bfit = solution[0]

    return afit,bfit,theta # values for fiting a spiral in exponential,polar coordinates form

def fit_spiral_opt(center_pre,x,y):
    """ Function to otpimize tge center location in order to minimize error
    Inputs : Estimated initial center (X-Y coordinates), Data point coordinates X, Data point coordinates Y
    Outputs : Optimized center location
    """
    # shift the other points with respect to the center location
    y = y - center_pre[1]
    x = x - center_pre[0]
    # convert them to polar form
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    theta = np.unwrap(theta) # correction for angle mapping
    solution = np.polyfit(theta,np.log(r),1)    # spiral : log(r) = log(a) + b*theta
    afit = np.exp(solution[1])
    bfit = solution[0]
    rfit = afit*np.exp(bfit*theta)
    return np.mean(abs(rfit-r)) # return the error in fit, this choice was for robustness

def find_center(pre, pre_binarized, draw):
    """ Function to estimate the center of the spiral of the cochlea through the use of Hugh circles
    Inputs: Pre-operative image, binarized image, Draw: Bool indicating wheter or not to return an image
    Oututs: If draw is True, function outputs an image with the circle and center drawn.
            If draw is False, function outputs the coordinates of the circle center.

    """
    # Convert the pre-operative image to color scale
    cimg = cv2.cvtColor(pre,cv2.COLOR_GRAY2BGR)

    pre_binarized = 255-pre_binarized

    # Calculate circular Hough transform on the binarized pre-operative image
    circles = cv2.HoughCircles(pre_binarized,cv2.HOUGH_GRADIENT,1,1000, param1=200,param2=15,minRadius=50,maxRadius=230)
    # Round the values
    circles = np.uint16(np.around(circles))

    # Draw the circles
    if draw == True :
        for i in circles[0,:]:
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        return cimg
    # Otherwise return points
    else :
        return circles


def blob_detection(post_masked_processed, draw, type=1):
    """ Funtion that performs blob detection to localize electrodes.

    Draw : If true, returns an image with the blobs drawn. If false, returns the blob coordinates
    Type : (1) Uses erosion and dilation. 2 Avoids erosion and dilation.
    """
    # Normalize image
    img = cv2.normalize(post_masked_processed, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype('uint8')

    # Apply median blurring
    img = cv2.medianBlur(img, 7)

    # Use sauvola thresholding to binarize the post operative image
    th2 = filters.threshold_sauvola(img)
    th2 = 255-th2
    th2 = th2.astype("uint8")

    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 10

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Apply erosion and dilation
    if type==1:
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(th2, kernel, iterations=2)
        dilation = cv2.dilate(erosion, kernel, iterations=6)
    # Or not
    elif type==2:
        dilation = th2

    # Detect blobs
    keypoints = detector.detect(dilation)

    # Either draw Blobs in the image as red circles
    if draw==True:
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(post_masked_processed, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        number_of_blobs = len(keypoints)
        text = "Number of Circular Blobs: " + str(len(keypoints))
        cv2.putText(post_masked_processed, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        return blobs

    # Or return blob coordinates
    else :
        result = []
        for point in keypoints:
            x = point.pt[0]
            y = point.pt[1]
            result.append([x,y])
        return result

def plot_spiral(opt_center, opt_spiral, img, save=False, saveDir=None, ID=None):
    '''
    Function to plot a spiral on an image
    '''

    thetas = np.arange(np.min(opt_spiral[2]),np.max(opt_spiral[2]),(np.max(opt_spiral[2])-np.min(opt_spiral[2]))/100) # simulate angles for smoothness
    rfitted = opt_spiral[0]*np.exp(opt_spiral[1]*thetas)
    # Plot spiral
    plt.plot()
    plt.imshow(img)
    plt.plot(opt_center[0]+rfitted*np.sin(thetas),opt_center[1]+rfitted*np.cos(thetas))
    plt.plot(opt_center[0],opt_center[1],"ro")
    if save == False :
    #ploted as converted into cartesian
        plt.show()
        plt.close()
    else :
        plt.savefig(saveDir+ID+"spiral3.jpg")
        plt.close()
    return 0


def get_labels(coordinates,center_pre):
    """
    Function to label electrodes from 1 to 12 based on blob center coordinates
    Inputs : Blob coordinates, center of cochlea
    Outputs : Labeled electrode coordinates from 1 to 12
    """
    #shift the coordinates with the respect to center
    coordinates = coordinates-center_pre
    # find the twelth electrode on the criteria of having one of the longest distances from center and the furthest distance from all other
    v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
    candidate_index = np.argpartition(-1*v_lengths,2)
    twelth = coordinates[candidate_index[:2],:] # get the furthest electrodes from center
    potential = np.zeros(2)
    for i in range(2):
        potential[i]=np.sum(np.sqrt(np.sum((coordinates-twelth[i])**2,axis=1)))
    candidate_index= np.where(coordinates==twelth[np.argmax(potential),:]) # get the electrode with highest euclid dist from the rest
    candidate = np.ndarray.flatten(coordinates[candidate_index[0][0],:])
    coordinates = np.delete(coordinates,candidate_index[0][0],axis=0)
    candidate_length = np.sqrt(np.sum(candidate**2))
    labels = [candidate]
    # at each step select the next electrode and pop it from the list
    # keep iterating until only one electrode is left
    while coordinates.shape[0]>1:
        v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
        angles = np.arccos(np.sum(candidate*coordinates,axis=1)/(candidate_length*v_lengths))
	# get the electrodes with the closest angular distance from the current electrode
        closest_angles = np.argpartition(angles,min(3,coordinates.shape[0]-1))
        angle_idx = closest_angles[:min(3,coordinates.shape[0])]
        candidates = coordinates[angle_idx,:]
        if coordinates.shape[0]>5: # vector lengths have been more effective for electrodes in the outer part
            closest = candidates[np.argmin(np.abs(candidate_length-v_lengths[angle_idx])),:]
        else:
	    # for the innermost electrodes euclid works better
            closest = candidates[np.argmin(np.sqrt(np.sum((candidates-candidate)**2,axis=1))),:]
        candidate_index = np.where(coordinates==closest)
        candidate = coordinates[candidate_index[0][0]]
        coordinates = np.delete(coordinates,candidate_index[0][0],axis=0)
        candidate_length = np.sqrt(np.sum(candidate**2))
        labels.append(candidate)
    labels.append(np.ndarray.flatten(coordinates))
    return labels

def insertion_depth2(coordinates,center_pre):
    '''
    Function to calculate the insertion depth of electrodes
    Inputs : coordinates of electrodes and the center position
    Outputs : angle values,sorted from the 12th towards the 1st electrode and their respective indices in coordinate array
    '''

    # determine position according to the center reference point
    coordinates = coordinates-center_pre
    # estimate the line going from electrode 12 through the origin
    slope = (coordinates[0][1])/(coordinates[0][0])
    intercept =center_pre[1] -  center_pre[0]*slope
    # get their angle by the dotproduct calculation
    v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))

    p = np.arccos(np.sum(coordinates[0,:]*coordinates,axis=1)/(v_lengths[0]*v_lengths))
    coordinates+= center_pre
    p[0] = 0
    # shift mapping to go from 0 to 2pi
    p2 = np.copy(p) # avoiding unintentional mutations
    for values in range(len(p)):
        # determine the electrode position with respect to the specified line
        y_estimate = slope*coordinates[values][0] + intercept
        if y_estimate > coordinates[values][1]: # if above the line shift by appropriate radians
            p2[values] = 2*np.pi- p2[values]
    # switch to degrees and round
    p2 = np.degrees(p2)
    p2 = np.round(p2)
    # find the position where angles decrease,indicating a shift by 360 might be required
    for i in range(len(p2)-1):
        if p2[i+1]<p2[i]:
            if p2[i]>=200: # from the data we concluded that angles do not go that far away
                break
    lower = np.where(p2[i+1:]<p2[i]) # get all the electrodes from given point that should be shifted
    if lower[0].size!=0:
       lower+=np.array(i+1)
       p2[lower]+=360 # shift the angles and preform sorting
    # this also corrects some misplaced labels
    label = np.argsort(p2)
    return p2[label],label


def blob_prune(blobs, center):
    '''
    Function to remove false blobs that are too far a way from the center to be part of the cochlear implant
    Inputs : Blob coordinates, Center coordinates
    Outputs : Blob coordinates with false positives removed
    '''
    # Avoid any copying problems
    newblobs = blobs.copy()
    distances = np.ndarray(shape=(len(blobs)))

    # Calculate euclidean distance between each blob and the center of the spiral
    for b in range(len(blobs)):
        distances[b] = np.linalg.norm(center-blobs[b])

    # Calculate the median distance
    median_distance = np.median(distances)

    # If the distance of the blob is over 3*the median distance, remove it·
    for b in range(len(distances)):
        if distances[b] > 3*median_distance:
            newblobs.remove(newblobs[b])
    return newblobs

def infer_points(center_opt,coordinates):
    '''
    Function to infer innermost electrodes
    Inputs : coordinates of the final inner electrodes and the center position
    Outputs: coordinates of the missing electrode with respect to center
    '''
    # fit a log spiral to the electrodes in coordinates
    x,y=coordinates[:,0],coordinates[:,1]
    y = y - center_opt[1]
    x = x - center_opt[0]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    theta = np.unwrap(theta) # correction for angle mapping
    solution = np.polyfit(theta,np.log(r),1)    # spiral : log(r) = log(a) + b*theta
    afit = np.exp(solution[1])
    bfit = solution[0]
    rfit = afit*np.exp(bfit*theta)
    deltas = []
    # infer the average distance between electrodes in form of arc length with constant factors removed
    for i in range(1,len(rfit)):
        deltas.append(abs(rfit[i]-rfit[i-1]))
    deltal = np.mean(np.array(deltas))
    # infer the radius vector of the missing electrode
    restimate = rfit[-1]-deltal
    # get its angular value as well
    thetaestimate = np.log(restimate/afit)/bfit
    # switch the coordinates back to cartesian form
    return np.sin(thetaestimate)*restimate,np.cos(thetaestimate)*restimate
