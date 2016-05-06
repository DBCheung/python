import cv2
import numpy as np
import math

"""
gesture recognition using skin_color convexity defects
"""
def main():
    cap = cv2.VideoCapture(0)
    while(True):
        if cap.isOpened() == False:
            print 'Failed to open camera device!'
            break 
        ret, img = cap.read()
    
        cv2.rectangle(img, (100,100), (300,300), (0,255,0))     # draw a rectangle: 2 points, line color
        crop_img = img[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        GaussFilterSize = (35, 35)
        blurred = cv2.GaussianBlur(grey, GaussFilterSize, 0)
        # first argu must be grayscale, second is thres_value, 3rd is max_val to be
        # assigned if pixel value is larger than thres_value, 4th is type
        _, thresh1 = cv2.threshold(blurred, 50, 255,
                                   cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #            cv2.THRESH_BINARY_INV, 13, 2)
        cv2.imshow('Thresholded', thresh1)
        # 1st argu is src img, findcontours() will modify src img, so make a copy;
        # 2nd argu is contour retrieval mode;
        # 3rd argu is contour approximation method;
        # contours is list of contours, each contour is np array of (x,y) coordinates,
        # CHAIN_APPROX_NONE stores all the coordinates
        _, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, \
                cv2.CHAIN_APPROX_NONE)
        max_area = -1
        for i in range(len(contours)): # find contour with max area
            maxAreaCnt=contours[i]
            area = cv2.contourArea(maxAreaCnt)
            if(area>max_area):
                max_area=area
                ci=i
        maxAreaCnt=contours[ci]
        x,y,w,h = cv2.boundingRect(maxAreaCnt)
        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
        hull = cv2.convexHull(maxAreaCnt)
        cnt_ConvHull = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(cnt_ConvHull, [maxAreaCnt], 0, (0,255,0), 0)    # draw max_area contour
        cv2.drawContours(cnt_ConvHull, [hull], 0, (0,0,255), 0)          # draw convex hull
        hull = cv2.convexHull(maxAreaCnt,returnPoints = False)
        defects = cv2.convexityDefects(maxAreaCnt,hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)       # draw all the contours
        # compute count_defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(maxAreaCnt[s][0])
            end = tuple(maxAreaCnt[e][0])
            far = tuple(maxAreaCnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57    # turn radian into angle
            if angle <= 80:
                count_defects += 1
                # prototype: cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                cv2.circle(crop_img, far, 3, [0,0,255], -1)
            # prototype: cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
            cv2.line(crop_img, start, end, [0,255,0], 2)    # mark the defects
        # prototype: cv2.putText(img, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) 
        # org: bottom left corner of the text string
        if count_defects == 1:
            cv2.putText(img, "1 finger", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
            cv2.putText(img, "2 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 3:
            cv2.putText(img, "3 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img, "4 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "5 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        cv2.imshow('Gesture Recognizer', img)
        combine_img = np.hstack((cnt_ConvHull, crop_img))
        cv2.imshow('LEFT: Contours, RIGHT: CROP_IMG', combine_img)
        
        # wait for 50ms untill next frame
        k = cv2.waitKey(50)
        if k == 27:
            break

if __name__ == '__main__':
    main()

