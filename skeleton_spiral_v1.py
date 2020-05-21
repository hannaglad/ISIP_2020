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
    import matplotlib.pyplot as plt
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()
    return 0

def find_center(img):
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = 255-th2
    circles = cv2.HoughCircles(th2,cv2.HOUGH_GRADIENT,1,1000,
                                param1=200,param2=15,minRadius=50,maxRadius=230)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    return circles

def searchSpace(pre, post):
    pre_color = cv2.cvtColor(pre,cv2.COLOR_GRAY2BGR)
    ret2,th2 = cv2.threshold(pre,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = 255-th2
    circles = cv2.HoughCircles(th2,cv2.HOUGH_GRADIENT,1,1000,
                            param1=200,param2=15,minRadius=50,maxRadius=230)
    circles = np.uint16(np.around(circles))
    x,y,r = circles[0,:][0]
    rows, cols = pre.shape

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

def skel(pre, post):

    post = searchSpace(pre, post)

    # Normalize
    post = cv2.normalize(post, None, 0, 255, cv2.NORM_MINMAX)

    # Blur
    #post = cv2.GaussianBlur(post, (11,11),4)
    post = np.uint8(post)

    #display(post)
    # Binarize
    post = cv2.bilateralFilter(post, 11, 700, 700)
    ret2,th2 = cv2.threshold(post,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th2 = cv2.adaptiveThreshold(post, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 201,2)
    #display(th2)
    #th2 = cv2.GaussianBlur(th2, (3,3),5)
    #th2 = invert(th2)

    # Opening
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=3)

    # current erosion5, dilation20
    dilation = cv2.dilate(erosion, kernel, iterations=28)
    th2 = cv2.normalize(dilation, None, 0, 1, cv2.NORM_MINMAX)
    th2 = cv2.GaussianBlur(th2, (21,21),5)



    # Save binarization results
    #th2 = th2*255
    #cv2.imwrite(dataDir+'result/skel/binar/'+ID+'.png', th2)
    #th2 = th2/255

    # skeletonize
    skeleton = skeletonize(th2)
    skeleton = skeleton*255
    skeleton = skeleton.astype(int)
    return skeleton

def get_skel_coordinates(skeleton):
    coordinates = np.transpose(np.nonzero(skeleton==255))
    return coordinates

def findElectrodes(pre, post):
    post_reduced = searchSpace(pre, post)
    # Find contours in post operative image
    post_reduced = cv2.normalize(post_reduced, None, 0, 255, cv2.NORM_MINMAX)

    post_reduced = np.uint8(post_reduced)
    post_reduced = cv2.bilateralFilter(post_reduced, 9, 75, 75)

    ret2,th2 = cv2.threshold(post_reduced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=15)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    im, cnts, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find spiral center in pre opertive image
    pre = find_center(pre)
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

def all(dataDir):
    for file in os.listdir(dataDir):

        if "result" not in file:
            ID = str(file[2:])
            path = dataDir+file+'/'

            if "post" in os.listdir(path)[0] :
                post = os.listdir(path)[0]
                pre = os.listdir(path)[1]
            else :
                pre = os.listdir(path)[0]
                post = os.listdir(path)[1]

            # LOAD DATA
            pre = path+pre
            post = path+post
            pre = cv2.imread(pre, 0)
            post = cv2.imread(post, 0)

            # INITIAL PREPROCESSING
            pre = cv2.normalize(pre,  None, 0, 255, cv2.NORM_MINMAX)
            pre = cv2.medianBlur(pre,13)

            # SKELETONIZE AND GET COORDINATES of skeleton
            skeleton = skel(pre, post)
            coordinates = get_skel_coordinates(skeleton)
            center_pre = find_center(pre)[0,0,0:2]

            # Get coordinates of contour centers
            contour_centers = findElectrodes(pre, post)
            # FIT SPIRAL
            we = fit_spiral(center_pre, coordinates[:,1], coordinates[:,0])
            thetas = np.arange(np.min(we[2]),np.max(we[2]),(np.max(we[2])-np.min(we[2]))/100) # simulate angles for smoothness
            rfitted = we[0]*np.exp(we[1]*thetas) # simulate length

            plt.plot()
            plt.imshow(post)
            # ploted as converted into cartesian
            plt.plot(center_pre[0]+rfitted*np.sin(thetas),center_pre[1]+rfitted*np.cos(thetas))
            name = "spirals/spiral"+ID+".jpg"
            plt.savefig(name)
            plt.close()
            print(ID)
    return 0


dataDir = './Handout/DATA/'
all(dataDir)
