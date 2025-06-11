import cv2
import numpy as np

#this can detect my lips
cam = cv2.VideoCapture(0)
template = cv2.imread('template.jpg',cv2.IMREAD_GRAYSCALE)
#frame = cv2.imread('template1.jpg')

w,h = template.shape[::-1]

#print(frame)
#print('t',template)

while True:
    ret,frame = cam.read()

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >=0.9)

    for pt in zip(*loc):
        cv2.rectangle(img,pt,(pt[0] + w,pt[1]+h),(0,255,0),3)
        

    

    cv2.imshow('image',img)
    cv2.imshow('result',result)
    #print(result)

    if cv2.waitKey(1) == ord('q'):
        break
'''
n = int(input(">>>"))

if n == 1:
    cv2.imwrite('template.jpg',img)
'''    
cam.release()
cv2.destroyAllWindows()

