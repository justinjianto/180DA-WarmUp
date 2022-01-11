
import cv2
import numpy as np
 
cap = cv2.VideoCapture(0)
 
while 1:
    ret,frame =cap.read()
    # ret will return a true value if the frame exists otherwise False
    into_hsv =cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # changing the color format from BGr to HSV
    # This will be used to create the mask
    L_limit=np.array([98,50,50]) # setting the blue lower limit
    U_limit=np.array([139,255,255]) # setting the blue upper limit
        
 
    b_mask=cv2.inRange(into_hsv,L_limit,U_limit)
    # creating the mask using inRange() function
    # this will produce an image where the color of the objects
    # falling in the range will turn white and rest will be black
    contours, _ = cv2.findContours(b_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key = cv2.contourArea)

        (x, y, w, h) = cv2.boundingRect(max_contour)
        box = [x,y, x+w,y+h]
        cv2.rectangle(frame, (x,y+h), (x+w,y), (255, 0, 0), 2)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
 
cv2.destroyAllWindows()