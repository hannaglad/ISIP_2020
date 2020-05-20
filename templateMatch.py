import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import (measure, transform, feature, filters, color, draw)
import cv2
import imutils
import os
import math
from scipy.spatial.distance import pdist, squareform
import angle_draft as shit

dataDir = './Handout/DATA/'
pre = dataDir+'ID06/ID06pre.png'
post = dataDir+'ID06/ID06post.png'
templateDir = './electrodes/'

def display(image):
    import matplotlib.pyplot as plt
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()

def find_center(img):
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = 255-th2
    circles = cv2.HoughCircles(th2,cv2.HOUGH_GRADIENT,1,1000,
                                param1=200,param2=15,minRadius=50,maxRadius=230)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
    # draw the outer circlre
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    return circles

def findElectrodes(pre, post_reduced, post):
    # Find contours in post operative image
    post_reduced = cv2.normalize(post_reduced, None, 0, 255, cv2.NORM_MINMAX)
    #post_reduced = cv2.GaussianBlur(post_reduced, (11,11),4)

    post_reduced = np.uint8(post_reduced)
    post_reduced = cv2.bilateralFilter(post_reduced, 11, 700, 700)

    ret2,th2 = cv2.threshold(post_reduced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th2 = cv2.adaptiveThreshold(post_reduced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,1)

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=13)
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
        centers.append([cX, cY])
        cv2.drawContours(post, c, -1, (0,255,0),1)
        cv2.circle(post, (cX, cY),3, (0,255,0), -1)
    #return cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
    return centers


    #return X,Y
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
            if math.hypot(i-x, j-y) > r+350:
                mask[j,i] = mean_value
            else :
                mask[j,i] = post[j,i]


    return mask


def templateMatch(post, nb):
    # Find contours in post operative image
    #post= cv2.normalize(post, None, 0, 255, cv2.NORM_MINMAX)

    # Get template image
    template_name = 'electrode_'+str(nb)+'.png'

    template = cv2.imread('electrodes/'+template_name, 0)

    w, h = template.shape[::-1]

    method = cv2.TM_SQDIFF_NORMED

    result = cv2.matchTemplate(post, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = min_loc
    bottom_right = (top_left[0]+w, top_left[1]+h)
    cv2.rectangle(post, top_left, bottom_right, 255, 2)

    position = (top_left[0]+w-30, top_left[1]+h-10)
    cv2.putText(post, str(nb), position, cv2.FONT_HERSHEY_SIMPLEX, 1,(209,80,255), 3)

    center = [top_left[0]+w/2, top_left[1]+h/2]

    return center

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


def coordinate_extract(template_centers):
    # convert annotated electrodes to ints
    for i in template_centers:
        i[1] = int(i[1])
    # sort them according to their numbering
    template_centers.sort(key=lambda center: center[1])
    return [i[0] for i in template_centers]   # get coordinates of the centers 


post = cv2.imread(post, 0)
pre = cv2.imread(pre, 0)
pre = cv2.normalize(pre,  None, 0, 255, cv2.NORM_MINMAX)
pre = cv2.medianBlur(pre,13)

template_centers = []

for template_file in os.listdir(templateDir):

    nb = [str(template_file[l]) for l in range(len(template_file)) if template_file[l].isdigit() == True]
    nb = ''.join(nb)

    template_center = templateMatch(post, nb)
    template_centers.append([template_center, nb])
center_pre = find_center(pre)[0,0,0:2]
coordinates = np.array(coordinate_extract(template_centers))
ins_d = insertion_depth(coordinates, center_pre)
mask = searchSpace(pre,post)
centers = findElectrodes(pre, mask, post)


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

we = fit_spiral(center_pre, coordinates[:9,0], coordinates[:9,1])
thetas = np.arange(np.min(we[2]),np.max(we[2]),(np.max(we[2])-np.min(we[2]))/100) # simulate angles for smoothness
rfitted = we[0]*np.exp(we[1]*thetas)
plt.imshow(post)

# ploted as converted into cartesian
plt.plot(center_pre[1]+rfitted*np.sin(thetas),center_pre[0]+rfitted*np.cos(thetas))
plt.show()