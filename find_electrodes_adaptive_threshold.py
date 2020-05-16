def findElectrodes(pre, post):
    # Find contours in post operative image
    post = cv2.imread(post,0)
    pre = cv2.imread(pre, 0)

    post = cv2.normalize(post, None, 0, 255, cv2.NORM_MINMAX)
    post = cv2.GaussianBlur(post, (11,11),4)
    #display(post)
    #ret2,th2 = cv2.threshold(post,0,255,cv2.THRESH_BINARY)

    th2 = cv2.adaptiveThreshold(post, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,1)

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
