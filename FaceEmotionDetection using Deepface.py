#Face Emotion Detection using Deep Neural Network. Shorter, faster, more optimized code than FER (Deep Neural Network).
#Better version of PR #48

import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
#File location to be inserted between ''
img = cv2.imread('')
plt.imshow(img[:,:,::-1])
plt.show()

result = DeepFace.analyze(img,actions=['emotion'])

print(result)
