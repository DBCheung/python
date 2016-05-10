#coding=utf-8

import cv2
import numpy as np
import math

"""
gesture recognition using skin color and convexity defects
TODOs: first background detection and subtraction!!!!!
"""
def main():
    cap = cv2.VideoCapture(0)
    while(True):
        if cap.isOpened() == False:
            print 'Failed to open the camera device 0!'
            break 
        
        ret, img = cap.read()
        cv2.rectangle(img, (100,100), (300,300), (0,255,0))     # draw a ROI rectangle: 2 points, line color
        crop_img = img[100:300, 100:300]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (25, 25), 0)
       


        ## 使用hand detect并进行分割，而不是threshold
        # detect hand using skin color 


        # backgroud subtraction using codebook




        # first argu must be grayscale, second is thres_value, 3rd is max_val to be
        # assigned if pixel value is smaller than thres_value, 4th is type
        _, thresh1 = cv2.threshold(blurred, 80, 255,
                                   cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #            cv2.THRESH_BINARY_INV, 13, 2)
        # cv2.imshow('gray image', gray)
        # cv2.imshow('Thresholded', thresh1)
        # 1st argu is src img, findcontours() will modify src img, so make a copy;
        # 2nd argu is contour retrieval mode;
        # 3rd argu is contour approximation method;
        # contours is list of contours, each contour is a list of np array of (x,y) coordinates,
        # CHAIN_APPROX_NONE stores all the coordinates
        _, contours, _= cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                cv2.CHAIN_APPROX_NONE)
        
        # find the contour with max_area
        max_area = -1
        for i in range(len(contours)): 
            maxAreaCnt = contours[i]
            area = cv2.contourArea(maxAreaCnt)
            if(area > max_area):
                max_area=area
                ci=i
        maxAreaCnt=contours[ci]

#        # mark the max_area contour with a red rectangle in crop_img
#        x,y,w,h = cv2.boundingRect(maxAreaCnt)
#        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
        
        # compute convex hull of max_area contour and its convexity defects
        hull = cv2.convexHull(maxAreaCnt)
        cnt_ConvHull = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(cnt_ConvHull, [maxAreaCnt], 0, (0,255,0), 0)    # draw max_area contour
        cv2.drawContours(cnt_ConvHull, [hull], 0, (0,0,255), 0)          # draw convex hull
        hull = cv2.convexHull(maxAreaCnt,returnPoints = False)
        defects = cv2.convexityDefects(maxAreaCnt, hull)
        cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)            # draw all the contours
        count_defects = 0
        # compute count_defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(maxAreaCnt[s][0])
            end = tuple(maxAreaCnt[e][0])
            far = tuple(maxAreaCnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57         # turn radian into angle
            if angle <= 80 and (b > 20 or c > 20) :
                count_defects += 1
                # prototype: cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                cv2.circle(crop_img, far, 3, [0,0,255], -1)              # mark the defects
            # prototype: cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
            cv2.line(crop_img, start, end, [0,255,0], 2)

        # recognition
        # prototype: cv2.putText(img, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) 
        # org: bottom left corner of the text string
        if count_defects == 1:
            cv2.putText(img, "1 finger",  (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
            cv2.putText(img, "2 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 3:
            cv2.putText(img, "3 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "4 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 5:
            cv2.putText(img, "5 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "No finger", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

        gray_thresh = np.hstack((gray, thresh1))
        cv2.imshow('Gesture Recognizer', img)
        cv2.imshow('LEFT to RIGHT: Gray, Thresh', gray_thresh)
        cv2.imshow('MaxCnt&Hull', cnt_ConvHull)
        
        # wait for N ms untill next frame
        k = cv2.waitKey(100)
        if k == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()

