
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters, measure
import cv2
import imutils

dataDir = './Handout/DATA/'
## Crop the images to remove the label
testImage = dataDir+'ID38/ID38pre.png'
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

def scaleImage(image):
    '''
    Use min max scaling to scale image to the range 0-255
    Converts to float values

    Current Issue : Lines appear on image (I predict future problems)
    '''
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1), copy=False)
    scaledImage = scaler.fit_transform(image)
    return scaledImage

def display(image):
    import matplotlib.pyplot as plt
    plt.plot()
    plt.imshow(image, cmap='gray')
    #plt.show()
    #plt.close()


cropped = cropImage(image, 50)
scaled = scaleImage(cropped)

# Remove noise using a gausian 5x5 kernel
blurred = filters.gaussian(scaled, sigma=5)

#thresh = filters.threshold_minimum(blurred)
#blurred = blurred > thresh
#blurred = blurred.astype(int)

# Contour detection
contours = measure.find_contours(blurred, 0.45)
display(blurred)
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()
