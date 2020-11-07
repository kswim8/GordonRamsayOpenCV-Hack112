import numpy as np
import cv2
import pickle
import time, random
from playsound import playsound
import cmu_112_graphics

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

inactiveCount = 0

def timerFired(app):
    app.timerDelay = 10
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
    print(faces)
    for (x, y, w, h) in faces:
    	print(x, y, w, h)
    	if ((x == None) and (y == None) and (w == None) and (h == None)):
    		inactiveCount += 1
    		if inactiveCount > 30:
    			#play sound
    			playsound('gordonramsay.mp3')
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

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=4 and conf <= 85:
    		#print(5: #id_)
    		#print(labels[id_])
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, '',(x,y), font, 1, color, stroke, cv2.LINE_AA)
		
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

    	img_item = "7.png"
    	cv2.imwrite(img_item, roi_color)

    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    	#subitems = smile_cascade.detectMultiScale(roi_gray)
    	#for (ex,ey,ew,eh) in subitems:
    	#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
