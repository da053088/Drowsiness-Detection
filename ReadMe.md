# NTUST 人工智慧與邊緣運算實務_期末專題報告_M11015Q18_羅大祐
## 作品名稱 
【智慧交通】- 疲勞駕駛偵測
## 摘要
隨著汽車的數量逐漸增多，交通事故發生的頻率也逐漸上升。發生交通事故的原因很多，其中疲勞駕駛佔交通事故原因的14%-20%。希望透過偵測駕駛的疲勞程度，降低因疲勞駕駛發生事故的機率。

## 系統簡介

### 創作發想
疲勞駕駛造成的車禍數不勝數。台灣每年因駕駛分心或疲勞駕駛而發生事故比例約占車禍總事故的20％，居各類事故原因第二名。
此作品打算利用人臉辨識去分辨駕駛是否出現過於疲勞這一現象。
希望可以達到警示駕駛的目的。
### 硬體架構
CPU： Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz   2.71 GHz
GPU：Intel(R) HD Graphics 620 / 8G
作業系統： win 10  

### 工作原理及流程
1. 由 Webcam 讀取 real-time 影像
2. 計算 EAR(Eye Aspect Ratio，眼睛縱橫比)
3. 如果EAR低於閥值連續30幀，判定為Drowsiness(有睡意)
![](https://i.imgur.com/25GsNyt.png)
![](https://i.imgur.com/RrS9cXq.png)


### 資料集建立方式
使用資料集ibug 300w
會對每個image的人臉標註68個特徵點

![](https://i.imgur.com/qUCASWD.png)
![](https://i.imgur.com/H0jOHYz.png)

### 模型選用與訓練
在local端用的是Dlib，要先把環境建起來
```
pip install dlib
pip install python

```
如果安裝Dlib遇到困難，請參考[這篇](https://blog.csdn.net/qq_42951560/article/details/116110382)

接著就可以下載source code
```
git clone https://github.com/chitgyi/Driver-Drowsiness-Detection-and-Alert-System.git
cd Driver-Drowsiness-Detection-and-Alert-System
```
下載[iBUG](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz) Datasets 並解壓，然後將解壓後的文件夾重命名為 datasets
打開train_eye.py
更改第62行路徑到你自己的datasets路徑
![](https://i.imgur.com/xiH81xf.png)
接下來執行
```
python parse_eye.py
python train_eye.py
```
執行後就能看到訓練好的.dat檔
![](https://i.imgur.com/nW44Xpd.png)

### 實驗結果
安裝幾個必要套件
```
pip install opencv-python
pip install imutils
```
建立一個.py檔，命名為mydetect並打開

導入必要的python libraries
```
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
```
宣告一個回傳EAR的函數
```
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear
```
更改環境變數
FACIAL_LANDMARK_PREDICTOR ： 你訓練好的.dat名稱
MINIMUM_EAR ： EAR 閥值 建議設定0.25~0.2之間。此 EAR 不是單眼的，而是雙眼的累積 EAR 平均值。
MAXIMUM_FRAME_COUNT ：EAR 的值變化非常快。即使眨眼，EAR 也會迅速下降。 所以這個變量告訴了 EAR 可以保持小於MINIMUM_EAR的最大連續幀數，否則會提醒睡意。
```
FACIAL_LANDMARK_PREDICTOR = "eye_predictor.dat"  
MINIMUM_EAR = 0.25
MAXIMUM_FRAME_COUNT = 30
```

取的dlib參數並開啟webcam
```
faceDetector = dlib.get_frontal_face_detector()
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
webcamFeed = cv2.VideoCapture(0)
```
找到兩隻眼睛 id 的開始值和結束值。
![](https://i.imgur.com/H6MvV6n.png)

```
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
```

逐幀讀取webcam，抓取左右眼的座標值，計算EAR取平均
記下EAR低於閥值的次數，達成條件(連續30幀)便印出Drowsiness
```
EYE_CLOSED_COUNTER = 0
try:
    while True:
        (status, image) = webcamFeed.read()
        image = imutils.resize(image, width=800)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceDetector(grayImage, 0)

        for face in faces:
            faceLandmarks = landmarkFinder(grayImage, face)
            faceLandmarks = face_utils.shape_to_np(faceLandmarks)

            leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
            rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
            cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)

            if ear < MINIMUM_EAR:
                EYE_CLOSED_COUNTER += 1
            else:
                EYE_CLOSED_COUNTER = 0

            cv2.putText(image, "EAR: {}".format(round(ear, 1)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                cv2.putText(image, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", image)
        cv2.waitKey(1)
except:
    pass
```    
一切就緒後執行
```
python myDetect.py
```
![](https://i.imgur.com/B0NvTP0.jpg)
![](https://i.imgur.com/K12Tpz7.jpg)



## 改進與優化
可改進的地方：
EAR的大小與個人的眼睛大小有著不小的關連，只用此來判斷相對不嚴謹
當側臉時(只拍攝到單眼)，會因為抓不到雙眼資料而計算不出EAR
優化方式：
增加多種判定方式，目前只使用EAR進行判定，未來可增加像是眨眼頻率、點頭次數等多種可判定為有睡意的邊準。


## 結論
目前模型已經可以初步判定是否合眼過久，不過想以此就推定有睡意稍顯不足。
## 參考文獻
https://github.com/chitgyi/Driver-Drowsiness-Detection-and-Alert-System
https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
https://blog.csdn.net/qq_42951560/article/details/116110382
https://www.cnblogs.com/wuliytTaotao/p/14594178.html
## 附錄
[Colab源碼](https://colab.research.google.com/drive/1OGqMMGwmd07cFM6aFz57qQiX9Iv33pso)
需先將訓練好的.dat檔以及想測試的影片片段放置在自己的雲端硬碟的目錄中
如果沒有訓練好的.dat檔，可先[下載](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)

