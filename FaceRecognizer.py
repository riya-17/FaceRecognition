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
from scipy.ndimage import zoom

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

#rotation of the image
def rotate_image(image, angle):
    #Size of the image input
    size = (image.shape[1], image.shape[0])
    center_of_image = tuple(np.array(size) / 2)
    rotation = np.vstack(
        [cv2.getRotationMatrix2D(center_of_image, angle, 1.0), [0, 0, 1]]
    )

    rotation2 = np.matrix(rotation[0:2, 0:2])
    width = size[0] * 0.5
    height = size[1] * 0.5
    rotated_coordinates = [
        (np.array([-width,  height]) * rotation2).A[0],
        (np.array([ width,  height]) * rotation2).A[0],
        (np.array([-width, -height]) * rotation2).A[0],
        (np.array([ width, -height]) * rotation2).A[0]
    ]

    # Size of the image output
    x_coordinates = [pt[0] for pt in rotated_coordinates]
    x_positif = [x for x in x_coordinates if x > 0]
    x_negatif = [x for x in x_coordinates if x < 0]
    y_coordinates = [pt[1] for pt in rotated_coordinates]
    y_positif = [y for y in y_coordinates if y > 0]
    y_negatif = [y for y in y_coordinates if y < 0]
    right = max(x_positif)
    left = min(x_negatif)
    top = max(y_positif)
    bottom = min(y_negatif)
    new_width = int(abs(right - left))
    new_height = int(abs(top - bottom))
    translation_matrix = np.matrix([
        [1, 0, int(new_width * 0.5 - width)],
        [0, 1, int(new_height * 0.5 - height)],
        [0, 0, 1]
    ])

    compute_matrix = (np.matrix(translation_matrix) * np.matrix(rotation))[0:2, :]
    result = cv2.warpAffine(
        image,
        compute_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR
    )
    return result

#Zooming in and out on an a given image (use less than 1 to zoom out/ more than 1 to zoom in/ 1 will keep the same zoom power as the original)
def zoom_in_out(image, zoom_power):
    heights, widths = image.shape[:2]
    zoom_tuple = (zoom_power,) * 2 + (1,) * (image.ndim - 2)

    # Zooming in
    if zoom_power > 1:
        zoom_heights = int(np.round(heights / zoom_power))
        zoom_widths = int(np.round(widths / zoom_power))
        top_zoom_in = (heights - zoom_heights) // 2
        left_zoom_in = (widths - zoom_widths) // 2
        out = zoom(image[top_zoom_in:top_zoom_in+zoom_heights, left_zoom_in:left_zoom_in+zoom_widths], zoom_tuple)
    
    # Zooming out
    elif zoom_power < 1:
        zoom_heights = int(np.round(heights * zoom_power))
        zoom_widths = int(np.round(widths * zoom_power))
        top_zoom_out = (heights - zoom_heights) // 2
        left_zoom_out = (widths - zoom_widths) // 2
        out = np.zeros_like(image)
        out[top_zoom_out:top_zoom_out+zoom_heights, left_zoom_out:left_zoom_out+zoom_widths] = zoom(image, zoom_tuple)

    # If zoom_power == 1 then do nothing and return the input image
    else:
        out = image
    return out


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
	"shape_predictor": "./shape_predictor_68_face_landmarks.dat",
	"image": "./examples/8.jpg",
        "encodings": "./encodings/encodings.pickle",
        "detection_method": "cnn"

}

# initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
face = FaceAligner(predictor, desiredFaceWidth=256)

# Load input image, resize and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Rotate Image 
rot_image = rotate_image(image, 180)
cv2.imshow("Image Rotated 90", rot_image)

#Zoom in & out Image
zoom_image = zoom_in_out(image, 1.3)
cv2.imshow("zoom", zoom_image)

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
    cv2.waitKey(0)

# show output with facial landmarks
cv2.imshow("Landmarks", image)

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
cv2.imshow("Detected face", image)
cv2.waitKey(0)
