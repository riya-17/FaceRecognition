# Capture image from Laptop Camera
import cv2

def Capture():
	cam = cv2.VideoCapture(0)
	
	cv2.namedWindow("Camera")
	
	img_counter = 0
	
	while True:
	    ret, frame = cam.read()
	    cv2.imshow("test", frame)
	    if not ret:
	        break
	    k = cv2.waitKey(1)
	
	    if k%256 == 27:
	        # ESC pressed
	        print("Escape hit, closing...")
	        break
	    elif k%256 == 32:
	        # SPACE pressed
	        img_name = "{}.png".format(img_counter)
	        cv2.imwrite("Captured/{}".format(img_name), frame)
	        print("{} written!".format(img_name))
		img_counter += 1
	
	cam.release()
	
	cv2.destroyAllWindows()