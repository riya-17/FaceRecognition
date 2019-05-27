# Face Recognizer
Face Recognizer is about Recognizing Faces of an Individual with the help of their facial features. <br><br>
Facial Recognizer uses deep learning algorithms to compare a live capture or digital image with the stored faceprints(also known as datasets) to verify an identity.<br><br>
The Algorithm used for classification is [k-NN model i.e. k-Nearest Neighbor classifier](https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/). It uses Euclidean distance to compare images for similarity. <br><br>  

# Prerequisites

Build and install dlib library

```
git clone https://github.com/davisking/dlib.git
mkdir build
cd build
cmake ..
cmake --build
cd ..
python setup.py install
```


# Setup
```
pip install -r requirements.txt
```

Set the path of the input images in the code and run the following command:
```
python FaceRecognizer.py
```

# How it Works?
* Image from which Face needs to be Recognized is loaded i.e. Input Image.
* The face is detected and cropped.
* if the face is not Aligned Straight then it is aligned.
* Landmarks are detected i.e. 68 (x, y)-coordinates that map to facial structures on the face.
* The Detected Face is encoded i.e. 128-d embeddings of the image are created.
* The input image[encoding] is passed to k-NN model for classification.
* k-NN model return the name with the highest votes 
<br><br>
# Outputs:

* Input Image - Image in which Faces are to be recognized:

![Input Image](https://user-images.githubusercontent.com/25060937/43034709-54d27858-8cff-11e8-8247-2a92cc6e4119.PNG)<br><br>

* Original Image - Image consists of cropped original Image

![Original Image](https://user-images.githubusercontent.com/25060937/43034712-6430b9cc-8cff-11e8-87f6-3a8926a46570.PNG)     ![Original Image](https://user-images.githubusercontent.com/25060937/43034730-a0e970fc-8cff-11e8-8cf4-0c137d9cc445.PNG)!<br><br>

* Aligned Image - Aligning the image increases the efficiency if the Algorithm

![Aligned image](https://user-images.githubusercontent.com/25060937/43034725-940c2d52-8cff-11e8-9c83-803d93966a1e.PNG)     ![Aligned Image](https://user-images.githubusercontent.com/25060937/43034732-a36fade6-8cff-11e8-885e-6a2a84fe8ebc.PNG)<br><br>

* Landmarks - Shows the Landmarks of theDetected Faces

![Landmark](https://user-images.githubusercontent.com/25060937/43034737-b3bf7866-8cff-11e8-9f0c-7be8f4071ddb.PNG)<br><br>

* Detected Face - The Face is recognized and the name of the recognized face is displayed along with the face. If the face does not belong to the Dataset than the Face is tagged as Unknown.

![Detected face](https://user-images.githubusercontent.com/25060937/43034739-b58304f6-8cff-11e8-8e93-68cae1883b30.PNG)<br><br>
<br><br><br>
**Another Recognition:**
<br><br>
* Input Image:

![Input Image](https://user-images.githubusercontent.com/25060937/43034745-d6b98ece-8cff-11e8-99a5-ee06cc01447c.PNG)<br><br>

* Original Image:                             

![Original Image](https://user-images.githubusercontent.com/25060937/43034746-d7ec80ee-8cff-11e8-99f3-d2fbc9b0d408.PNG)<br><br>          

* Aligned Image:

![Aligned Image](https://user-images.githubusercontent.com/25060937/43034747-d9125926-8cff-11e8-81df-5661d1a4ead1.PNG)<br><br>

* Landmarks :

![Landmarks](https://user-images.githubusercontent.com/25060937/43034748-da4a4100-8cff-11e8-8b59-76cc7803e080.PNG)<br><br>

* Detected Face :

![Detected Face](https://user-images.githubusercontent.com/25060937/43034749-dba060ca-8cff-11e8-8f90-2dc4765f586c.PNG)<br><br>
<br>

### encode-faces.py

It is used to create 128-d face embeddings of the input image as well as custom dataset. These embeddings are used to compare input image(embeddings) with the dataset(embeddings), the one with the highest votes is preferred.<br><br>

### DAT file

[Link to Download Dat File](https://osdn.net/projects/sfnet_dclib/downloads/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/)<br><br>

# Resources

* I got big help from [pyimagesearch](https://www.pyimagesearch.com/pyimagesearch-gurus/). It cleared most of my concepts with very Ease and practical examples.
* [Adam Geitgey](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) : How Face Recognition works
* [Face Recognition](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
* [Custom Dataset](https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/): helped me to create my own dataset.
* [Must Read Post](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)
* [Deep Face Recognition](http://krasserm.github.io/2018/02/07/deep-face-recognition/)
* [Facial Landmarks](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
* [Facial Alignment](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
* [Face Recognition for Beginners](https://towardsdatascience.com/face-recognition-for-beginners-a7a9bd5eb5c2)
* [Face Recognition Basics](https://www.coursera.org/lecture/convolutional-neural-networks/what-is-face-recognition-lUBYU)
