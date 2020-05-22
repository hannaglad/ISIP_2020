import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import (measure, transform, feature, filters, color, draw)
import cv2
import imutils
import os
import math
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize
from skimage.util import invert

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
    #dilation = cv2.dilate(erosion, kernel, iterations=20)
    th2 = cv2.normalize(th2, None, 0, 1, cv2.NORM_MINMAX)
    th2 = cv2.GaussianBlur(th2, (21,21),5)

    # Save binarization results
    #th2 = th2*255
    #cv2.imwrite(dataDir+'result/skel/binar/'+ID+'.png', th2)
    #th2 = th2/255

    # skeletonize
    skeleton = skeletonize(th2)
    skeleton = skeleton*255
    skeleton = skeleton.astype('uint8')
    skel_dilate = cv2.dilate(skeleton, kernel, iterations=5)

    return skel_dilate

def blob_detection(post_processed):
    # Find contours in post operative image
    #th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,0)

    th2 = post_processed
    th2 = 255-th2
    params = cv2.SimpleBlobDetector_Params()

     # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 50

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    kernel = np.ones((5,5), np.uint8)

    # Erosion and dilation of binarized image
    erosion = cv2.erode(th2, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=6)
    #Detect blobs
    keypoints = detector.detect(dilation)
    coordinates = []
    for point in keypoints:
        x = point.pt[0]
        y = point.pt[1]
        #size = point.size
        coordinates.append([x,y])

    return coordinates

def get_skel_coordinates(skeleton):
    coordinates = np.transpose(np.nonzero(skeleton==255))
    return coordinates

if __name__=="__main__":

    # Load data
    dataDir = './Handout/DATA/'
    pre = dataDir+'ID55/ID55pre.png'
    post = dataDir+'ID55/ID55post.png'
    pre = cv2.imread(pre, 0)
    post = cv2.imread(post, 0)

    # Process images
    pre_binarized = binar_otsu(pre)
    post_processed = full_preprocess(post)
    circles = find_center(pre, pre_binarized, False)
    post_masked = searchSpace(pre_binarized, post, circles)
    post_masked_processed = full_preprocess(post_masked)

    # Find blobs and their centers in post operative image
    blobs = blob_detection(post_processed)

    # Skeletonize the post operative image to fit spiral
    skeleton = skel(pre, post_masked_processed)
    coordinates = get_skel_coordinates(skeleton)


    coordinates = np.vstack((coordinates, blobs))


    # Estimate spiral center in pre operative image
    center_pre = find_center(pre, pre_binarized, False)[0,0,0:2]

    # Fit spiral
    we = fit_spiral(center_pre, coordinates[:,1], coordinates[:,0])
    thetas = np.arange(np.min(we[2]),np.max(we[2]),(np.max(we[2])-np.min(we[2]))/100) # simulate angles for smoothness
    rfitted = we[0]*np.exp(we[1]*thetas) # simulate length

    # Plot spiral
    plt.plot()
    plt.imshow(post)
    #ploted as converted into cartesian
    plt.plot(center_pre[0]+rfitted*np.sin(thetas),center_pre[1]+rfitted*np.cos(thetas))
    plt.show()
    plt.close()
