# Face Recognizer(얼굴 인식기)
1) 설명 <br>
Face Recognizer은 ‘얼굴의 특성(이목구비)’으로 사람 얼굴을 식별하고, 결과를 이름으로 나타낸다.<br><br>

2) 결과 도출 방법 <br>
딥러닝 알고리즘을 사용하여 ‘방금 찍은 사진 이나 
디지털 이미지’를 데이터셋(faceprints)과 비교해서 결과 도출<br><br>

3) k-NN model 사용 <br>
분류를 위해서 [k-NN 모델 i.e. k-Nearest Neighbor classifier](https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/)을 사용하였고, 
이미지 유사성은 ‘유클리디안 거리’를 사용하여 비교하였다.<br><br>

# 사전 요구사항

Build and install dlib library (해당 repository clone 한곳에서 시작)

```
pip install cmake (cmake 없을 시)
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake ..
cmake --build
cd ..
python setup.py install
```


# 필요한 패키지 설치
#### Windows:
```
pip install -r requirements.txt
```
#### Linux:
- Setup environment
    ```
    sudo apt-get update
    sudo apt-get upgrade
    ```
- OpenCV를 위한 패키지 설치
    ```
    sudo apt-get install build-essential cmake unzip pkg-config
    sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
    sudo apt-get install libjasper-dev
    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
    sudo apt-get install libxvidcore-dev libx264-dev
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libatlas-base-dev gfortran
    sudo apt-get install python3.6-dev
    sudo apt-get install libboost-all-dev
    ```
- OpenCV 설치
    - OpenCV
        ```
        pip install opencv-python
        ```
    - OpenCV Contrib
        ```
        pip install opencv-contrib-python
        ```
- 기타 의존 패키지 설치
    ```pip install -r requirements.txt```
      
    ### 오류 발생 시 해결
    - On installing opencv:
        ```https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/```
    - On dlib:
        ```https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/```
    

코드에 input images의 path 경로 추가하고, 다음 명령어 실행:
```
python FaceRecognizer.py
```

# 동작 과정 설명
* 인식할 얼굴이 있는 이미지(e.g., input image)를 로드한다.
* 얼굴이 감지되고 분할된다.
* 얼굴이 바르게 align 되어있지 않으면 align 해준다.
* 얼굴 구조에서 Landmarks(e.g., 68 (x, y) 좌표) (=얼굴 특징점) 을 감지한다.
* 감지된 face landmarks를 인코딩 한다. (e.g., 이미지의 128-d 임베딩 생성).
* 인코딩 된 input image는 k-NN 모델을 통과하여 분류된다.
* k-NN 모델은 가장 높은 정확도의 이름을 반환해준다.
<br><br>
# 결과물:

* Input Image - 얼굴이 인식될 이미지:

![Input Image](https://user-images.githubusercontent.com/25060937/43034709-54d27858-8cff-11e8-8247-2a92cc6e4119.PNG)<br><br>

* Original Image - input image를 자른 것

![Original Image](https://user-images.githubusercontent.com/25060937/43034712-6430b9cc-8cff-11e8-87f6-3a8926a46570.PNG) <br>
![Original Image](https://user-images.githubusercontent.com/25060937/43034730-a0e970fc-8cff-11e8-8cf4-0c137d9cc445.PNG)!<br><br>

* Aligned Image - 이미지를 align하면 알고리즘의 효과가 높다.
![Aligned image](https://user-images.githubusercontent.com/25060937/43034725-940c2d52-8cff-11e8-9c83-803d93966a1e.PNG) <br>
![Aligned Image](https://user-images.githubusercontent.com/25060937/43034732-a36fade6-8cff-11e8-885e-6a2a84fe8ebc.PNG)<br><br>

* Landmarks - 감지된 얼굴의 landmarks(얼굴특징점)을 보여준다.

![Landmark](https://user-images.githubusercontent.com/25060937/43034737-b3bf7866-8cff-11e8-9f0c-7be8f4071ddb.PNG)<br><br>

* 감지된 얼굴 - 얼굴이 인식되고, 인식된 얼굴의 이름이 얼굴을 따라 표시된다. 얼굴이 데이터셋에 속하지 않았을 경우, 얼굴은 Unknown으로 라벨링 된다. 

![Detected face](https://user-images.githubusercontent.com/25060937/43034739-b58304f6-8cff-11e8-8e93-68cae1883b30.PNG)<br><br>
<br><br><br>
**또다른 인식 예시:**
<br><br>
* Input Image:

![Input Image](https://user-images.githubusercontent.com/25060937/43034745-d6b98ece-8cff-11e8-99a5-ee06cc01447c.PNG)<br><br>

* Original Image:                             

![Original Image](https://user-images.githubusercontent.com/25060937/43034746-d7ec80ee-8cff-11e8-99f3-d2fbc9b0d408.PNG)<br><br>          

* Aligned Image:

![Aligned Image](https://user-images.githubusercontent.com/25060937/43034747-d9125926-8cff-11e8-81df-5661d1a4ead1.PNG)<br><br>

* Landmarks :

![Landmarks](https://user-images.githubusercontent.com/25060937/43034748-da4a4100-8cff-11e8-8b59-76cc7803e080.PNG)<br><br>

* 감지된 얼굴 :

![Detected Face](https://user-images.githubusercontent.com/25060937/43034749-dba060ca-8cff-11e8-8f90-2dc4765f586c.PNG)<br><br>
<br>

### encode-faces.py
커스텀 데이터 셋 뿐만아니라 input image의 128 차원의 임베딩을 생성해준다. 이 임베딩은 input image(임베딩)과 데이터셋(임베딩)을 비교하는데 사용된다. 가장 높은 선택을 받은 것이 예측 결과가 된다. <br><br>

### DAT file

[Dat File 다운로드 링크](https://osdn.net/projects/sfnet_dclib/downloads/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/)<br><br>

# 도움 및 참고
* [pyimagesearch](https://www.pyimagesearch.com/pyimagesearch-gurus/) 에서 큰 도움을 받음. 매우 쉽고 실용적인 예제로 나의 대부분의 개념을 명확히 함.
* [Adam Geitgey](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) : Face Recognition의 동작 법
* [Face Recognition](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
* [Custom Dataset](https://www.pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/): 데이터 셋 생성할 수 있도록 도움.
* [Must Read Post](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)
* [Deep Face Recognition](http://krasserm.github.io/2018/02/07/deep-face-recognition/)
* [Facial Landmarks](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
* [Facial Alignment](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)
* [Face Recognition for Beginners](https://towardsdatascience.com/face-recognition-for-beginners-a7a9bd5eb5c2)
* [Face Recognition Basics](https://www.coursera.org/lecture/convolutional-neural-networks/what-is-face-recognition-lUBYU)
