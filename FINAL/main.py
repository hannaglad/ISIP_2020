import functions as f
import os
import cv2
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    ####### LOAD DATA #######

    # Set directories
    dataDir = './Handout/DATA/'
    saveDir = './results/'

    # For each ID
    for file in os.listdir(dataDir):

        # Define pre and post operative images
        ID = str(file[2:])
        # Define path
        path = dataDir+file+'/'

        if "post" in os.listdir(path)[0] :
            post = os.listdir(path)[0]
            pre = os.listdir(path)[1]
        else :
            pre = os.listdir(path)[0]
            post = os.listdir(path)[1]

        print("Currently processing ID ", ID)

        # Load images
        pre = path+pre
        post = path+post
        pre = cv2.imread(pre, 0)
        post = cv2.imread(post, 0)

        #########################
        ##### PREPROCESSING #####

        # Process pre operative image
        pre = cv2.normalize(pre,  None, 0, 255, cv2.NORM_MINMAX)
        pre = cv2.medianBlur(pre, 13)
        pre_binarized = f.binar_otsu(pre)

        # Process post operative image
        post_processed = f.full_preprocess(post)
        #########################
        #### MAIN ALGORITHM #####

        # Estimate spiral center
        circles = f.find_center(pre, pre_binarized, False)
        center_pre = circles[0,0,0:2]

        # Define Hough circle as inner search space
        post_masked = f.searchspaceMin(pre_binarized, post, circles)
        post_masked_processed = f.full_preprocess(post_masked)

        # Blob detection on whole post operative image
        blobs = f.blob_detection(post, False)

        # Blob detection on inner search space
        spiral_blobs = f.blob_detection(post_masked,False, type=1)

        # Use non opened images for blob detection on problematic images
        if len(blobs) < 12:
            # Blob detection on whole image
            blobs = f.blob_detection(post, False, type=2)
            # Blob detection on inner search space
            spiral_blobs = f.blob_detection(post_masked, False, type=2)

        # Remove eventual false positive electrodes
        blobs = f.blob_prune(blobs, center_pre)

        # Extract blob coordinates to fit spiral
        spiral_coords_full = np.vstack(spiral_blobs)

        # Fit an optimal spiral center through the blob coordinates
        opt_center = fmin(f.fit_spiral_opt,center_pre,args=(spiral_coords_full[:,0], spiral_coords_full[:,1]), disp=False) # minimize spiral error by finding the best center

        # Estimate initial labelling of electrodes
        hope = f.get_labels(blobs, center_pre) + center_pre
        # Calculate insertion depth
        insertionDepth,label = f.insertion_depth2(hope,opt_center)
        # Re-order electrodes
        hope = hope[label,:]
        # Infer missing electrodes
        while hope.shape[0]<12:
            infered = f.infer_points(opt_center,hope[hope.shape[0]-5:,:]) +  opt_center
            infered = infered[np.newaxis,:]
            hope = np.append(hope,infered,axis=0)


        # Raise warning if there is more 12 electrodes
        if hope.shape[0] != 12:
            print("{} electrodes detected in image {}. Results should not be considered".format(hope.shape[0], ID))

        # Re-calculate insertion depth
        insertionDepth,label = f.insertion_depth2(hope,opt_center)
        hope = hope[label,:]
        count = 0

        # Display electrode ordering on the post-operative image
        cv2.circle(post,(int(opt_center[0]),int(opt_center[1])),5,(0,0,255),3)
        for i in hope:
            x = i[0]
            y = i[1]
            x = int(x)
            y = int(y)
            cv2.putText(post, str(12-count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),2)
            cv2.circle(post,(x,y),2,(0,0,255),3)
            count+=1
        plt.imsave(saveDir+"images/"+ID+".jpg",post)

        with open(saveDir+"project_results.txt", 'a') as file:
            file.write("ID:{} | Angular Depth:{}".format(ID, insertionDepth))
            file.write("\n")
            file.write("Spiral center:{}". format(opt_center))
            file.write("\n")
            file.write("Electrode coordinates:{}".format(hope))
            file.write("\n")
            file.write("\n")
