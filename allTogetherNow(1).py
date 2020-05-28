import functionsC as f
import os
import cv2
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    # Load data
    dataDir = './Handout/DATA/'
    saveDir = './results/'
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

            print("Currently processing ID ", ID)

            pre = path+pre
            post = path+post
            pre = cv2.imread(pre, 0)
            post = cv2.imread(post, 0)

            # Process images
            pre = cv2.normalize(pre,  None, 0, 255, cv2.NORM_MINMAX)
            pre = cv2.medianBlur(pre, 13)
            pre_binarized = f.binar_otsu(pre)
            post_processed = f.full_preprocess(post)
            circles = f.find_center(pre, pre_binarized, False)

            ### CHANGE DIAMETER OF SEARCH SPACE BY COMMENTING OUT ####
            # Diameter of 150 around the hugh circle
            post_masked = f.searchSpace150(pre_binarized, post, circles)
            # Diameter of 75
            #post_masked = f.searchSpace75(pre_binarized, post, circles)
            # Just the hugh circle
            post_masked = f.searchspaceMin(pre_binarized, post, circles)
            ##########################################################
            post_masked_processed = f.full_preprocess(post_masked)

            # Find blobs and their centers in post operative image
            # First pass
            blobs = f.blob_detection(post, False)
            #blob_img = f.blob_detection(post, True)
            spiral_blobs = f.blob_detection(post_masked, False)

            # Second pass
            if len(blobs) < 12:
                blobs = f.blob_detection2(post, False)
                blob_img = f.blob_detection2(post, True)
                spiral_blobs = f.blob_detection2(post_masked, False)
                if len(blobs) < 12:
                    print("Less than 12 blobs detected in", ID)

            #cv2.imwrite(saveDir+"blobs/"+ID+".jpg", blob_img)
            # Skeletonize
            #skeleton = f.med_axis(post_masked_processed)

            # Extract coordinates from skeleton and blobs spiral fitting


            ########## COORDINATES TO USE FOR SPIRAL : COMMENT OUT ONE YOU DONT WANT TO USE #####
            # Spiral Coordinates and blobs
            #spiral_coords_full = np.vstack((f.get_skel_coordinates(skeleton), spiral_blobs))

            # Just the blobs
            spiral_coords_full = np.vstack(spiral_blobs)
            #####################################################

            # Estimate spiral center
            center_pre = f.find_center(pre, pre_binarized, False)[0,0,0:2]

            # Fit spiral
            opt_center = fmin(f.fit_spiral_opt,center_pre,args=(spiral_coords_full[:,0], spiral_coords_full[:,1]), disp=False) # minimize spiral error by finding the best center
            #opt_spiral = f.fit_spiral(opt_center, spiral_coords_full[:,0], spiral_coords_full[:,1])# take the best center and fit the spiral

            # Plot the spiralito
            #f.plot_spiral(opt_center, opt_spiral, pre, save=True, saveDir=saveDir, ID=ID)

            # Number and label electrodes

            hope = f.get_labels(blobs, center_pre) + center_pre 
            insertionDepth,label = f.insertion_depth2(hope,opt_center)
            hope = hope[label,:]
            count = 0
            for i in hope:
                x = i[0]
                y = i[1]
                x = int(x)
                y = int(y)
                cv2.putText(post, str(12-count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),2)
                count+=1
            plt.imsave(saveDir+ID+".jpg",post)

            with open("project_results.txt", 'a') as file:
                file.write("Image ID:{} | Angular Depth:{}".format(ID, insertionDepth))
                file.write("\n")
