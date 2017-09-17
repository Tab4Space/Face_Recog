import cv2, os
import tensorflow as tf
import numpy as np
from dataAugmentation import generate_image_data
from generateCSV import generate_CSV
from model import generateModel

BASEPATH = '/home/bhappy/Face_Recog/person/'


personImages = os.listdir('./person/')
personImages = sorted(personImages)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


## 디렉토리에서 이미지 불러오기 및 크롭핑
## 함수를 통해 원본데이터에서 데이터를 부풀림
for personImage in personImages:
    img_color = cv2.imread(BASEPATH + personImage, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    x, y, w, h = faces[0][0], faces[0][1], faces[0][2], faces[0][3]

    img_crop = img_color[y:y+h, x:x+w]              ## Image Cropping
    generate_image_data(img_crop, personImage)      ## 데이터 생성


generate_CSV()                                  ## CSV 파일 생성
generateModel()

