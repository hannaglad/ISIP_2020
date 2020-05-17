
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters, measure,feature
import cv2
import imutils
from scipy.signal import convolve2d

dataDir = '/home/milos/Desktop/ISIP_2020-master/Handout/DATA/'
## Crop the images to remove the label
testImage = dataDir+'ID55/ID55pre.png'
# Open the image and display a plot
image = imread(testImage, as_gray=True)


def cropImage(image, borderPixel):
    '''
    Crops image by removing all values before borderPixel, and all values after axis length-borderPixel
    '''
    # Would use the last value of the label, but all images need to be the same size
    lengthX = np.shape(image)[0]
    lengthY = np.shape(image)[1]
    croppedImage = image[borderPixel:lengthX-borderPixel, borderPixel:lengthY-borderPixel]
    return croppedImage


def display(image):
    import matplotlib.pyplot as plt
    plt.plot()
    plt.imshow(image, cmap='gray')
    #plt.show()
    #plt.close()






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
    plt.imshow(cimg)

    
img = cv2.imread(testImage,0)
img = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
img = cv2.medianBlur(img,13)
find_center(img)


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

def findElectrodes(pre, post_reduced, post):
    # Find contours in post operative image
    post_reduced = cv2.normalize(post_reduced, None, 0, 255, cv2.NORM_MINMAX)
    #post_reduced = cv2.GaussianBlur(post_reduced, (11,11),4)

    post_reduced = np.uint8(post_reduced)
    post_reduced = cv2.bilateralFilter(post_reduced, 9, 75, 75)
    #display(post_reduced)
    ret2,th2 = cv2.threshold(post_reduced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th2 = cv2.adaptiveThreshold(post_reduced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,0)

    #display(th2)

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=15)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    im, cnts, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find spiral center in pre opertive image
    pre = find_center(pre)
    post = cv2.cvtColor(post, cv2.COLOR_GRAY2BGR)
    #compute center of contours in post operative image
    X = []
    Y = []
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        X.append(cX)
        Y.append(cY)
        cv2.drawContours(post, c, -1, (0,255,0),1)
        cv2.circle(post, (cX, cY),3, (0,255,0), -1)
    return post
    #return X,Y
