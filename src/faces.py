import numpy as np
import cv2
import pickle
import time, random
from playsound import playsound

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

cap = cv2.VideoCapture(1)

inactiveCount = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    for (x, y, w, h) in faces:
    	print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]
		
    	startTime = time.time()
    	print(x, y, w, h)
    	(currentX, currentY, currentW, currentH) = (x, y, w, h)
		
    	if ((x == currentX) and (y == currentY) and (w == currentW) and (h == currentH)):
    		inactiveCount += 1
    		if inactiveCount > 30:
    			#play sound
    			sounds = ['are-you-always-this-pathetic.mp3','get-in-there.mp3', 'this-is-wrong.mp3']
    			# playsound('are-you-always-this-pathetic.mp3','get-in-there.mp3')
    			playsound(sounds[random.randint(0, len(sounds)-1)])
    			inactiveCount = 0
		
    	print(inactiveCount)

    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
