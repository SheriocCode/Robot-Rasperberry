#!/usr/bin/env python3

#导入所需的库
import cv2 as cv
import numpy as np
#检测函数
def face_detect(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #转化图像为灰度图
    face_detector = cv.CascadeClassifier("./data/haarcascade_frontalface_default.xml")#读取人脸数据
    faces = face_detector.detectMultiScale(gray,1.02,20)#进行人脸检测
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)#对人脸位置画框
    cv.imshow("face_detect", image)#展示
#运行人脸检测并显示
def video_face_detect():
    capture = cv.VideoCapture(0)#设置使用的相机
    while True:
        ret, frame = capture.read()#读取相机图像
        frame = cv.flip(frame, 1)#将回传画面设置图像水平翻转
        face_detect(frame)#人脸检测
        c = cv.waitKey(10)
        if c==27:  #按下ESC键退出
            break
 
if __name__ == '__main__':
    video_face_detect()#实时检测人脸
    cv.waitKey(0) #按下任意键退出
    cv.destroyAllWindows()
