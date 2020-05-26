import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import (measure, transform, feature, filters, color, draw)
import cv2
import imutils
import math
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize, medial_axis
from skimage.util import invert
from scipy.optimize import fmin

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

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    img = cv2.bilateralFilter(img, 11, 700, 700)
    img = cv2.GaussianBlur(img, (11,11),4)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def binar_otsu(image):
    """Performs otsu binarization on a grayscale image"""
    th2 = ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def searchSpace75(pre_binarized, post, circles):
    """Function to set region of image that does not contain electrodes to a mean of the image color
    Inputs : binarized image of pre-operative cochlea, post operative image of implant, and center the circle of implants
    Outputs : Image with the outside of the circle set to the mean image color
    """
    pre_binarized = 255-pre_binarized
    x,y,r = circles[0,:][0]
    rows, cols = post.shape

    mask = np.ones((post.shape[0], post.shape[1]))
    mean_value = np.mean(post)

    for i in range(cols):
        for j in range(rows):
            if math.hypot(i-x, j-y) > r+75: # originally 150
                mask[j,i] = mean_value
            else :
                mask[j,i] = post[j,i]

    return mask

def searchSpace150(pre_binarized, post, circles):
    """Function to set region of image that does not contain electrodes to a mean of the image color
    Inputs : binarized image of pre-operative cochlea, post operative image of implant, and center the circle of implants
    Outputs : Image with the outside of the circle set to the mean image color
    """
    pre_binarized = 255-pre_binarized
    x,y,r = circles[0,:][0]
    rows, cols = post.shape

    mask = np.ones((post.shape[0], post.shape[1]))
    mean_value = np.mean(post)

    for i in range(cols):
        for j in range(rows):
            if math.hypot(i-x, j-y) > r+150:
                mask[j,i] = mean_value
            else :
                mask[j,i] = post[j,i]

    return mask

def searchspaceMin(pre_binarized, post, circles):
    """Function to set region of image that does not contain electrodes to a mean of the image color
    Inputs : binarized image of pre-operative cochlea, post operative image of implant, and center the circle of implants
    Outputs : Image with the outside of the circle set to the mean image color
    """
    pre_binarized = 255-pre_binarized
    x,y,r = circles[0,:][0]
    rows, cols = post.shape

    mask = np.ones((post.shape[0], post.shape[1]))
    mean_value = np.mean(post)

    for i in range(cols):
        for j in range(rows):
            if math.hypot(i-x, j-y) > r: # originally 150
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
    # optimize tge center location to minimize error
    y = y - center_pre[1]
    x = x - center_pre[0]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    theta = np.unwrap(theta) # correction for angle mapping
    solution = np.polyfit(theta,np.log(r),1)    # spiral : log(r) = log(a) + b*theta
    afit = np.exp(solution[1])
    bfit = solution[0]
    rfit = afit*np.exp(bfit*theta)
    return np.mean(abs(rfit-r))

def find_center(pre, pre_binarized, draw):
    """ Function to estimate the center of the spiral of the cochlea through the use of Hugh circles
    Inputs: Pre-operative image, binarized image, Draw: Bool indicating wheter or not to return an image
    Oututs: If draw is True, function outputs an image with the circle and center drawn.
            If draw is False, function outputs the coordinates of the circle center.

    """
    cimg = cv2.cvtColor(pre,cv2.COLOR_GRAY2BGR)
    pre_binarized = 255-pre_binarized

    circles = cv2.HoughCircles(pre_binarized,cv2.HOUGH_GRADIENT,1,1000,
                                param1=200,param2=15,minRadius=50,maxRadius=230)
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

def skel(pre, post_masked_processed):
    """ Function to perform simple skeletonization on an image
    Inputs : Pre-operative image, post operative processed image
    Outputs : Skeletonized image
    """
    # Preprocess
    th2 = cv2.GaussianBlur(post_masked_processed, (17,17),5)
    th2 = cv2.normalize(th2, None, 0, 1, cv2.NORM_MINMAX)

    # skeletonize
    skeleton = skeletonize(th2)
    skeleton = skeleton*255
    skeleton = skeleton.astype('uint8')

    return skeleton

def blob_detection(post_masked_processed, draw):
    """ Funtion that performs blob detection to localize electrodes.
    Outputs : If draw is True, outputs an image with the blobs drawn
                  If draw is False, outputs the coordinates of the centers of the blobs
    """
    img = cv2.normalize(post_masked_processed, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype('uint8')
    img = cv2.medianBlur(img, 7)

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
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=6)

    # Detect blobs
    keypoints = detector.detect(dilation)

    # Save blob centers
    if draw==True:
        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(th2, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        number_of_blobs = len(keypoints)
        text = "Number of Circular Blobs: " + str(len(keypoints))
        cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        return blobs
    else :
        result = []
        for point in keypoints:
            x = point.pt[0]
            y = point.pt[1]
            result.append([x,y])
        return result

def blob_detection2(post_masked_processed, draw):
    """ Funtion that performs blob detection to localize electrodes. Does not include dilation and erosion. For use in problematic images where blobs are hard to locate.
    Inputs : Processed post oeprative image, draw (Bool)
    Outputs : If draw is True, outputs an image with the blobs drawn
                  If draw is False, outputs the coordinates of the centers of the blobs
    """

    # Initial set up; no erosion or dilationn

    img = cv2.normalize(post_masked_processed, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype('uint8')
    img = cv2.medianBlur(img, 11)

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
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(th2)

    if draw==True:
        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(th2, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        number_of_blobs = len(keypoints)
        text = "Number of Circular Blobs: " + str(len(keypoints))
        cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        return blobs
    else :
        result = []
        for point in keypoints:
            x = point.pt[0]
            y = point.pt[1]
            result.append([x,y])
        return result

def get_skel_coordinates(skeleton):
    """
    Function to return the coordinates where a skeletonized image is white
    Inputs : Skeletonized image
    Outputs : Coordinates where image is white
    """
    coordinates = np.transpose(np.nonzero(skeleton==255))
    return np.flip(coordinates, 1)

def med_axis(post_masked_processed):
    """
    Function to perform medial axis skeletonization on an image
    Input : Post operative processed image
    Outputs : Skeletonized image
    """
    th2 = cv2.GaussianBlur(post_masked_processed, (17,17),5)
    kernel = np.ones((3,3), np.uint8)

    th2 = cv2.erode(th2, kernel, iterations=5)
    th2 = cv2.dilate(th2, kernel, iterations=20)

    th2 = cv2.normalize(th2, None, 0, 1, cv2.NORM_MINMAX)
    skel, distance = medial_axis(th2, return_distance=True)

    distance_on_skel = ((distance*skel)*255).astype('uint8')
    ret2, distance_on_skel = cv2.threshold(distance_on_skel, 0, 255, cv2.THRESH_BINARY)

    return distance_on_skel

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
    coordinates = coordinates-center_pre
    v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
    candidate_index = np.argmax(v_lengths)
    candidate = coordinates[candidate_index,:]
    coordinates = np.delete(coordinates,candidate_index,axis=0)
    candidate_length = np.sqrt(np.sum(candidate**2))
    labels = [candidate]
    while coordinates.shape[0]>1:
        v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
        angles = np.arccos(np.sum(candidate*coordinates,axis=1)/(candidate_length*v_lengths))
        closest_angles = np.argpartition(angles,min(3,coordinates.shape[0]-1))
        angle_idx = closest_angles[:min(3,coordinates.shape[0])]
        candidates = coordinates[angle_idx,:]
        if coordinates.shape[0]>5:
            closest = candidates[np.argmin(np.abs(candidate_length-v_lengths[angle_idx])),:]
        else:
            closest = candidates[np.argmin(np.sqrt(np.sum((candidates-candidate)**2,axis=1))),:]
        candidate_index = np.where(coordinates==closest)
        candidate = coordinates[candidate_index[0][0]]
        coordinates = np.delete(coordinates,candidate_index[0][0],axis=0)
        candidate_length = np.sqrt(np.sum(candidate**2))
        labels.append(candidate)
    labels.append(np.ndarray.flatten(coordinates))
    return labels

def insertion_depth(coordinates,center_pre):
    """ Function to calculate the insertion depth of the cochlear implant
    Inputs : Coordinates of the labelled electrodes, center of the cochlea
    Outputs : Insertion depth of the cochlear implant
    """

    # determine position according to the center reference point
    coordinates = coordinates-center_pre
    # get their angle by the dotproduct calculation
    v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
    p = np.arccos(np.sum(coordinates[0,:]*coordinates,axis=1)/(v_lengths[0]*v_lengths))
    p[0] = 0
    # shift mapping from 0  1
    p2 = np.copy(p) # avoiding unintentional mutations
    for values in range(len(p)-1):
        if p[values]>p[values+1]:
            p2[values+1] = 2*np.pi- p2[values+1]
    # switch to degrees round and sort in reverse
    p2 = np.degrees(p2)
    p2 = np.round(p2)
    p2[np.argmax(p2)+1:] = p2[np.argmax(p2)+1:] + 360
    return p2
