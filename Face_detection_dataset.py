import cv2 #importing OpenCV file
import os #operation system library

camera = cv2.VideoCapture(0) #accessing the webcam using OpenCV

camera.set(3, 640) #width using for the images

camera.set(4, 480) #Height using for the images 

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #a provided machine learning algorithm that is used to identify the picture

face_name = input('\n Enter user name: ') #Asking user name

print("\n [INFO] Initializing face capture.") #status message

counter= 0 #countdown the nunber of pictures

while(True):
	ret,img=camera.read()

	gray_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRY)  #to convert the images to gray color

	faces = face_detector.detectMultiScale(gray_color, 1.3,5) #function used to detect the images

for(x, y, w,h) in faces:

	cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2) #to resize the images
	counter +=1

	cv2.imwrite("dataset_for_Face_Recognition/User." + str(face_name) + '.' + str(counter) + ".jpg", gray_color[y:y+h,x:x+w]) #save the images to the folder that I already create in my drive (must create your own folder and then update the correct path)
	cv2.imshow('image', img)

I = cv2.waitkey(100) & 0xff
if I ==47:
	break
elif counter >= 50:
	break    #taking 50 pictures 

print("\n [INFO] Existing program")

camera.release()

cv2.destroyAllWindows()
