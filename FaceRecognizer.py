# import the packages that are required

from imutils import face_utils
from imutils.face_utils import FaceAligner
import face_recognition
import numpy as np
import argparse
import imutils
import dlib
import pickle
import cv2
import uuid
import keyboard


def rotation(image,angle1):
	# rotate the image by 90 degrees
    rotated = imutils.rotate(image, angle=angle1)
    cv2.imshow("Rotated with Angle=%d" % (angle1), rotated)
    cv2.waitKey(0)
    return rotated

def rect_to_bb(rect):
    # we will take the bounding box predicted by dlib library
    # and convert it into (x, y, w, h) where x, y are coordinates
    # and w, h are width and height

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize (x, y) coordinates to zero
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop through 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)- coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# construct the arguments

# if you want to pass arguments at the time of running code
# follow below code and format for running code

"""
#ap.add_argument("-e", "--encodings", required=True,
#	help="path to serialized db of facial encodings")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#ap.add_argument("-d", "--detection-method", type=str, default="cnn",
#	help="face detection model to use: either `hog` or `cnn`")
#args = vars(ap.parse_args())

python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

"""

# if you want to use predefined path than define the path in a variable

args = {
	"shape_predictor":r"C:\Users\rishav kumar\FaceRecognition/shape_predictor_68_face_landmarks.dat",
	"image":r"C:\Users\rishav kumar\FaceRecognition\examples/1.jpg",
        "encodings":r"C:\Users\rishav kumar\FaceRecognition\encodings/encodings.pickle",
        "detection_method": "hog"
}

# initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
face = FaceAligner(predictor, desiredFaceWidth=256)

# Load input image, resize and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
cv2.imshow("Input", image)
rects = detector(gray, 1)
# loop over the faces that are detected
for (i, rect) in enumerate(rects):
    # Detected face landmark (x, y)-coordinates are converted into
    # Numpy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to OpenCV bounding box and draw
    # [i.e., (x, y, w, h)]
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = face.align(image, gray, rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


    f = str(uuid.uuid4())
    cv2.imwrite("foo/" + f + ".png", faceAligned)

    # shows the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y) coordinate for facial landmark and drow on th image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Original", faceOrig)
    cv2.imshow("Aligned", faceAligned)
    cv2.imshow("Landmarks", image)
    cv2.waitKey(1)

# show output with facial landmarks
cv2.imshow("Landmarks", image)
angle1=0
while True:
    if keyboard.is_pressed('r'):  # if key 'r' is pressed 
        print('Rotate by 90 degrees')
        angle1+=90
        image=rotation(image,angle1)
        cv2.waitKey(0)
        if angle1>=360:
            break
    elif keyboard.is_pressed('u'):
        # cv2.waitKey(0)
        print('zoom in')
        image = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("zoom in",image)
        cv2.waitKey(0)
    elif keyboard.is_pressed('l'):
    #cv2.waitKey(0)
        print('zoom out')
        height, width = image.shape[:2]
        image = cv2.resize(image,None,fx=0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("zoom out",image)
        cv2.waitKey(0)
    #elif on_press(key):
        #break
    else:
        pass



        
    

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y) coordinates of the bounding box corresponding to
# each face inthe input image and compute facial embeddings for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model = args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names of detected faces
names = []

# loop over facial embeddings
for encoding in encodings:
    # compares each face in the input image to our known encodings
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    # check if  match is found or not
    if True in matches:
        #find the indexes of all matches and initialize a dictionary
        # to count number of times a match occur
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over matched indexes and maintain a count for each face
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # Select the recognized face with maximum number of matches and
        # if there is a tie Python selects first entry from the dictionary
        name = max(counts, key=counts.get)

    # update the list of names
    names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
    # draw predicted face name on image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

# Output Image
angle1=0
cv2.imshow("Detected face", image)
#cv2.waitKey(0)
angle2=0
while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('r'):  # if key 'r' is pressed 
            print('Rotate by 90 degrees final')
            angle2+=90
            image=rotation(image,angle2)
            if angle2>360:
                break
        elif keyboard.is_pressed('u'):
           # cv2.waitKey(0)
            print('zoom in')
            image = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
            cv2.imshow("zoom in",image)
            cv2.waitKey(0)
        elif keyboard.is_pressed('l'):
           # cv2.waitKey(0)
            print('zoom out')
            height, width = image.shape[:2]
            image = cv2.resize(image,None,fx=0.5,fy=0.5, interpolation = cv2.INTER_CUBIC)
            cv2.imshow("zoom out",image)
            cv2.waitKey(0)
        else:
            cv2.waitKey(0)
            pass
    except:
        break


   





    












