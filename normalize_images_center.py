
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



def findElectrodes(pre, post):
    # Find contours in post operative image
    post = cv2.imread(post,0)
    pre = cv2.imread(pre, 0)

    post = cv2.normalize(post, None, 0, 255, cv2.NORM_MINMAX)
    post = cv2.medianBlur(post, 13)
    ret2,th2 = cv2.threshold(post,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = th2-255
    display(th2)

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=11)
    dilation = cv2.dilate(erosion, kernel, iterations=5)
    im, cnts, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find spiral center in pre opertive image
    pre = find_center(pre)
    post = cv2.cvtColor(post, cv2.COLOR_GRAY2BGR)
    #compute center of contours in post operative image
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        cv2.drawContours(post, c, -1, (0,255,0),1)
        cv2.circle(post, (cX, cY),3, (0,255,0), -1)
    display(post)
