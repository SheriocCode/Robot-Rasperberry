import sys
import cv2
import time
import math
import numpy as np

rect = None
size = (320, 240)
rotation_angle = 0
unreachable = False
world_X, world_Y = 0, 0

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    

  
# 设置检测的目标颜色
__target_color = ('red', 'green', 'blue')

#目标颜色阈值
range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255)}

# 颜色阈值
lab_data_max = {'red':  (255, 255, 255), 'black': ( 89, 255, 255), 'blue': ( 255, 254, 90), 'green': ( 255, 120, 180), 'white': ( 255, 255, 255)}
lab_data_min = {'red':  (0, 160, 135), 'black': ( 0, 0, 0), 'blue': ( 0, 120, 0), 'green': ( 0, 0, 100), 'white': ( 193, 0, 0)}

count = 0
_stop = False
color_list = []
get_roi = False
__isRunning = False
move_square = False
detect_color = 'None'
start_pick_up = False

# 变量重置
def reset(): 
    global _stop
    global count
    global get_roi
    global color_list
    global move_square
    global detect_color
    global start_pick_up
    
    count = 0
    _stop = False
    color_list = []
    get_roi = False
    move_square = False
    detect_color = 'None'
    start_pick_up = False

# 开启玩法
def start():
    global __isRunning
    reset()
    __isRunning = True
    print("ColorDetect Start")


# 找出面积最大的轮廓
# 参数为要比较的轮廓的列表
def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # 历遍所有轮廓
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 300:  # 只有在面积大于300时，最大面积的轮廓才是有效的，以过滤干扰
                area_max_contour = c

    return area_max_contour, contour_area_max  # 返回最大的轮廓

#坐标转换
def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


t1 = 0
roi = ()
center_list = []
last_x, last_y = 0, 0
draw_color = range_rgb["black"]
# 图像处理
def run(img):
    global roi
    global rect
    global count
    global get_roi
    global move_square
    global center_list
    global unreachable
    global __isRunning
    global start_pick_up
    global last_x, last_y
    global rotation_angle
    global world_X, world_Y
    global start_count_t1, t1
    global detect_color, draw_color, color_list
    
    if not __isRunning:
        return img
    else:
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]

        frame_resize = cv2.resize(img_copy, size, interpolation=cv2.INTER_NEAREST)#图像缩放
        frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)#高斯滤波
        
        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)  # 将图像转换到LAB空间

        color_area_max = None
        max_area = 0
        areaMaxContour_max = 0
        if not start_pick_up:
            for i in lab_data_max:
                if i in __target_color:
                    frame_mask = cv2.inRange(frame_lab,
                                                   (lab_data_min[i][0],
                                                  lab_data_min[i][1],
                                                  lab_data_min[i][2]),
                                                 (lab_data_max[i][0],
                                                  lab_data_max[i][1],
                                                  lab_data_max[i][2]))          #对原图像和掩模进行位运算，二值化
                    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 开运算
                    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算
                    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出轮廓
                    areaMaxContour, area_max = getAreaMaxContour(contours)  # 找出最大轮廓
                    if areaMaxContour is not None:
                        if area_max > max_area:  # 找最大面积
                            max_area = area_max
                            color_area_max = i
                            areaMaxContour_max = areaMaxContour
            if max_area > 500:  # 有找到面积大于500的轮廓
                (center_x, center_y), radius = cv2.minEnclosingCircle(areaMaxContour_max)  # 获取最小外接圆
                center_x = int(map(center_x, 0, size[0], 0, img_w)) #圆心x坐标
                center_y = int(map(center_y, 0, size[1], 0, img_h)) #圆心y坐标
                radius = int(map(radius, 0, size[0], 0, img_w))  #半径
                cv2.circle(img, (int(center_x), int(center_y)), int(radius), range_rgb[color_area_max], 2) #画出最小外接圆
                
                if not start_pick_up:
                    if color_area_max == 'red':  # 红色最大
                        color = 1
                    elif color_area_max == 'green':  # 绿色最大
                        color = 2
                    elif color_area_max == 'blue':  # 蓝色最大
                        color = 3
                    else:
                        color = 0
                    color_list.append(color)
                    if len(color_list) == 3:  # 多次判断
                        # 取平均值
                        color = int(round(np.mean(np.array(color_list))))
                        color_list = []
                        start_pick_up = True
                        if color == 1:
                            detect_color = 'red'
                            draw_color = range_rgb["red"]
                        elif color == 2:
                            detect_color = 'green'
                            draw_color = range_rgb["green"]
                        elif color == 3:
                            detect_color = 'blue'
                            draw_color = range_rgb["blue"]
                        else:
                            detect_color = 'None'
                            draw_color = range_rgb["black"]
            else:
                if not start_pick_up:
                    draw_color = (0, 0, 0)
                    detect_color = "None"  
        
        cv2.putText(img, "Color: " + detect_color, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, draw_color, 2)  #在回传画面打印下识别信息
        
        return img

if __name__ == '__main__':
    start()
    cap = cv2.VideoCapture(0)     #开启摄像头
    while True:
        # 移除了这一行，不再每次循环都重置start_pick_up
        # start_pick_up = False
        ret, img = cap.read()       #读取一帧摄像头图像，ret为布尔值。
        if ret:
            frame = img.copy() 
            Frame = run(frame)      
            frame_resize = cv2.resize(Frame, (320, 240))     
            cv2.imshow('frame', frame_resize)              #显示回传画面
            key = cv2.waitKey(1)
            if key == 27:  # ESC键退出
                break
        else:
            time.sleep(0.01)
    cv2.destroyAllWindows()
