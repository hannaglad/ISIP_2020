import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import (measure, transform, feature, filters, color, draw)
import cv2
import imutils
import os
import math
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize, medial_axis
from skimage.util import invert
from scipy.optimize import fmin

def rmse(vec1, vec2):
    """takes in 2 vectors and calculates the root mean square error"""
    diff = vec1-vec2
    return np.sqrt(np.dot(diff, diff)/len(vec1))

def display(image):
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()
    return 0

def full_preprocess(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    img = cv2.bilateralFilter(img, 11, 700, 700)
    img = cv2.GaussianBlur(img, (11,11),4)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def binar_otsu(image):
    th2 = ret2,th2 = cv2.threshold(pre,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def searchSpace(pre_binarized, post, circles):

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

def fit_spiral(center_pre,x,y):
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
    #return rmse(rfit,r)
    return np.mean(abs(rfit-r))

def find_center(pre, pre_binarized, draw):
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


    #th2 = cv2.adaptiveThreshold(post, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 201,2)
    #display(th2)
    th2 = cv2.GaussianBlur(post_masked_processed, (17,17),5)
    #th2 = invert(th2)

    # Opening
    kernel = np.ones((3,3), np.uint8)
    #erosion = cv2.erode(th2, kernel, iterations=10)

    # current erosion5, dilation20
    #dilation = cv2.dilate(erosion, kernel, iterations=8)
    #th2 = erosion

    th2 = cv2.normalize(th2, None, 0, 1, cv2.NORM_MINMAX)
    #th2 = cv2.GaussianBlur(th2, (21,21),5)

    # Save binarization results
    #th2 = th2*255
    #cv2.imwrite(dataDir+'result/skel/binar/'+ID+'.png', th2)
    #th2 = th2/255

    # skeletonize
    skeleton = skeletonize(th2)
    skeleton = skeleton*255
    skeleton = skeleton.astype('uint8')
    #skel_dilate = cv2.dilate(skeleton, kernel, iterations=5)

    return skeleton

def blob_detection(post, draw):
    img = cv2.normalize(post, None, 0, 255, cv2.NORM_MINMAX)
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

# Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(th2, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Save blob centers
    if draw==True:
        return blobs
    else :
        result = []
        for point in keypoints:
            x = point.pt[0]
            y = point.pt[1]
            result.append([x,y])
        return result

def get_skel_coordinates(skeleton):
    coordinates = np.transpose(np.nonzero(skeleton==255))
    return np.flip(coordinates, 1)

def findElectrodes(post, post_reduced_processed):

    th2 = post_reduced_processed.astype('uint8')

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=15)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    im, cnts, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    post = cv2.cvtColor(post, cv2.COLOR_GRAY2BGR)
    #compute center of contours in post operative image
    centers = []
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        centers.append([cX,cY])
        cv2.drawContours(post, c, -1, (0,255,0),1)
        cv2.circle(post, (cX, cY),3, (0,255,0), -1)
    return centers

def med_axis(post_masked_processed):
    th2 = cv2.GaussianBlur(post_masked_processed, (17,17),5)
    kernel = np.ones((3,3), np.uint8)

    th2 = cv2.erode(th2, kernel, iterations=5)
    th2 = cv2.dilate(th2, kernel, iterations=20)

    th2 = cv2.normalize(th2, None, 0, 1, cv2.NORM_MINMAX)
    skel, distance = medial_axis(th2, return_distance=True)

    distance_on_skel = ((distance*skel)*255).astype('uint8')
    ret2, distance_on_skel = cv2.threshold(distance_on_skel, 0, 255, cv2.THRESH_BINARY)

    return distance_on_skel


def get_labels(coordinates,center_pre):
    coordinates = coordinates-center_pre
    candidate_index = np.argmax(coordinates[:,0])
    candidate = coordinates[candidate_index,:]
    coordinates = np.delete(coordinates,candidate_index,axis=0)
    v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
    candidate_length = np.sqrt(np.sum(candidate**2))
    labels = [candidate]
    while coordinates.shape[0]>1:
        v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
        angles = np.arccos(np.sum(candidate*coordinates,axis=1)/(candidate_length*v_lengths))
        closest_angles = np.argpartition(np.abs(angles),min(3,coordinates.shape[0]-1))
        angle_idx = closest_angles[:min(3,coordinates.shape[0])]
        candidates = coordinates[angle_idx,:]
        closest = candidates[np.argmin(np.sqrt(np.sum((candidates-candidate)**2,axis=1))),:]
        candidate_index = np.where(coordinates==closest)
        candidate = coordinates[candidate_index[0][0]]
        coordinates = np.delete(coordinates,candidate_index[0][0],axis=0)
        candidate_length = np.sqrt(np.sum(candidate**2))
        labels.append(candidate)
    labels.append(np.ndarray.flatten(coordinates))
    return labels

def insertion_depth(coordinates,center_pre):


    # determine position according to the center reference point
    coordinates = coordinates-center_pre
    # get their angle by the dotproduct calculation
    v_lengths = np.sqrt(np.sum(coordinates**2,axis=1))
    p = np.arccos(np.sum(coordinates[11]*coordinates,axis=1)/(v_lengths[11]*v_lengths))
    # shift mapping from -1 1
    p2 = np.copy(p) # avoiding unintentional mutations
    for values in range(len(p)-1,0,-1):
        if p[values]>p[values-1]:
            p2[values-1] = 2*np.pi- p2[values-1]
    # switch to degrees round and sort in reverse
    p2 = np.degrees(p2)
    return np.round(p2[::-1])

if __name__=="__main__":

    # Load data
    dataDir = './Handout/DATA/'
    pre = dataDir+'ID17/ID17pre.png'
    post = dataDir+'ID17/ID17post.png'
    pre = cv2.imread(pre, 0)
    post = cv2.imread(post, 0)

    # Process images
    pre_binarized = binar_otsu(pre)
    post_processed = full_preprocess(post)
    circles = find_center(pre, pre_binarized, False)
    post_masked = searchSpace(pre_binarized, post, circles)
    post_masked_processed = full_preprocess(post_masked)

    # Find blobs and their centers in post operative image
    blobs = blob_detection(post, False)
    blob_img = blob_detection(post, True)
    display(blob_img)

    # Skeletonize the post operative image to fit spiral
    #skeleton = med_axis(post_masked_processed)
    #coordinates = get_skel_coordinates(skeleton)
    #coordinates = np.vstack((coordinates,blobs))

    coordinates = np.vstack(blobs)
    #coordinates = blobs
    # Estimate spiral center in pre operative image
    center_pre = find_center(pre, pre_binarized, False)[0,0,0:2]

    # Fit spiral
    p = fmin(fit_spiral_opt,center_pre,args=(coordinates[:,0], coordinates[:,1])) # minimize spiral error by finding the best center
    we = fit_spiral(p, coordinates[:,0], coordinates[:,1])# take the best center and fit the spiral
    thetas = np.arange(np.min(we[2]),np.max(we[2]),(np.max(we[2])-np.min(we[2]))/100) # simulate angles for smoothness
    rfitted = we[0]*np.exp(we[1]*thetas)

    # Plot spiral
    plt.plot()
    plt.imshow(blob_img)
    #ploted as converted into cartesian
    plt.plot(p[0]+rfitted*np.sin(thetas),p[1]+rfitted*np.cos(thetas))
    plt.plot(p[0],p[1],"ro")
    plt.show()
    plt.close()

    labels = get_labels(coordinates, center_pre) + center_pre
    count=0
    for i in labels:
        print(i)
        x = i[0]
        y = i[1]
        x = int(x)
        y = int(y)
        cv2.putText(blob_img, str(12-count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
        count+=1
    display(blob_img)
