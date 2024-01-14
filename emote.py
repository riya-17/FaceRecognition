from __future__ import division
from imutils import face_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import face_recognition
import argparse
import cv2
import os
import numpy as np

# Input parameters: already cropped color image of single face
# Output: detected emotion label
def detect_emotion(face_image):
	print("EMOTE!")

	cv2.imshow("EMOTE,EMOTE!", face_image)
	cv2.waitKey(0)

	# Open and read network model from json file
	file = open('emotionModel/fer.json', 'r')
	json = file.read()
	file.close()
	model = model_from_json(json)

	# Load weights into model
	model.load_weights("emotionModel/fer.h5")
	print("Loaded model")

	# Image parameters
	WIDTH = 48
	HEIGHT = 48
	x = None
	y = None
	labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	
	# Prepare image
	grayscale = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
	expanded_face = np.expand_dims(np.expand_dims(cv2.resize(grayscale, (48, 48)), -1), 0)
	cv2.normalize(expanded_face, expanded_face, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

	# Predict emotion
	yhat = model.predict(expanded_face)
	print("Emotion: "+ labels[int(np.argmax(yhat))])

	return labels[int(np.argmax(yhat))]