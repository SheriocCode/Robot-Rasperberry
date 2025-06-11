import cv2
import numpy as np
import sys

# 颜色定义
COLORS = {
    'red': {'hsv_min': (0, 120, 70), 'hsv_max': (10, 255, 255), 'rgb': (0, 0, 255)},
    'green': {'hsv_min': (35, 120, 70), 'hsv_max': (85, 255, 255), 'rgb': (0, 255, 0)},
    'blue': {'hsv_min': (100, 120, 70), 'hsv_max': (130, 255, 255), 'rgb': (255, 0, 0)},
    'yellow': {'hsv_min': (20, 120, 70), 'hsv_max': (30, 255, 255), 'rgb': (0, 255, 255)},
    'purple': {'hsv_min': (130, 120, 70), 'hsv_max': (160, 255, 255), 'rgb': (255, 0, 255)}
}

# 全局变量
selected_color = 'red'
detection_enabled = False
show_mask = False
use_roi = False
roi = None
contour_area_threshold = 5000  # 最小轮廓面积阈值
detection_history = []
history_size = 5  # 用于平滑检测结果的历史记录大小

# 创建窗口和滑动条
def create_trackbars():
    cv2.namedWindow('Trackbars')
    
    # 为每个颜色创建滑动条
    for color in COLORS:
        cv2.createTrackbar(f'{color}_h_min', 'Trackbars', COLORS[color]['hsv_min'][0], 179, lambda x: None)
        cv2.createTrackbar(f'{color}_s_min', 'Trackbars', COLORS[color]['hsv_min'][1], 255, lambda x: None)
        cv2.createTrackbar(f'{color}_v_min', 'Trackbars', COLORS[color]['hsv_min'][2], 255, lambda x: None)
        cv2.createTrackbar(f'{color}_h_max', 'Trackbars', COLORS[color]['hsv_max'][0], 179, lambda x: None)
        cv2.createTrackbar(f'{color}_s_max', 'Trackbars', COLORS[color]['hsv_max'][1], 255, lambda x: None)
        cv2.createTrackbar(f'{color}_v_max', 'Trackbars', COLORS[color]['hsv_max'][2], 255, lambda x: None)
    
    # 面积阈值滑动条
    cv2.createTrackbar('Area Threshold', 'Trackbars', contour_area_threshold, 50000, lambda x: None)
    
    # 颜色选择下拉菜单
    cv2.createTrackbar('Color Select', 'Trackbars', 0, len(COLORS)-1, on_color_select)

# 颜色选择回调函数
def on_color_select(val):
    global selected_color
    selected_color = list(COLORS.keys())[val]

# 鼠标回调函数，用于选择ROI
def select_roi(event, x, y, flags, param):
    global roi, use_roi
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x, y)
        use_roi = True
    elif event == cv2.EVENT_LBUTTONUP and use_roi:
        roi = (roi[0], roi[1], x, y)
        use_roi = False

# 从滑动条获取当前颜色的HSV阈值
def get_current_color_thresholds():
    h_min = cv2.getTrackbarPos(f'{selected_color}_h_min', 'Trackbars')
    s_min = cv2.getTrackbarPos(f'{selected_color}_s_min', 'Trackbars')
    v_min = cv2.getTrackbarPos(f'{selected_color}_v_min', 'Trackbars')
    h_max = cv2.getTrackbarPos(f'{selected_color}_h_max', 'Trackbars')
    s_max = cv2.getTrackbarPos(f'{selected_color}_s_max', 'Trackbars')
    v_max = cv2.getTrackbarPos(f'{selected_color}_v_max', 'Trackbars')
    
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

# 处理红色的特殊情况（在HSV空间中跨越0度）
def create_red_mask(hsv_image, h_min, s_min, v_min, h_max, s_max, v_max):
    if h_min > h_max:  # 红色跨越0度
        lower_red1 = np.array([h_min, s_min, v_min])
        upper_red1 = np.array([179, s_max, v_max])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        
        lower_red2 = np.array([0, s_min, v_min])
        upper_red2 = np.array([h_max, s_max, v_max])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        return cv2.bitwise_or(mask1, mask2)
    else:
        lower_red = np.array([h_min, s_min, v_min])
        upper_red = np.array([h_max, s_max, v_max])
        return cv2.inRange(hsv_image, lower_red, upper_red)

# 主处理函数
def process_frame(frame):
    global detection_enabled, show_mask, roi, contour_area_threshold, detection_history
    
    # 获取当前面积阈值
    contour_area_threshold = cv2.getTrackbarPos('Area Threshold', 'Trackbars')
    
    # 复制原始帧用于显示
    output_frame = frame.copy()
    
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 如果选择了ROI，只处理ROI区域
    if roi and len(roi) == 4:
        x1, y1, x2, y2 = roi
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 在输出帧上绘制ROI
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 处理ROI区域
        roi_hsv = hsv[y1:y2, x1:x2]
    else:
        roi_hsv = hsv
    
    # 获取当前颜色的阈值
    hsv_min, hsv_max = get_current_color_thresholds()
    
    # 创建颜色掩码
    if selected_color == 'red':
        mask = create_red_mask(roi_hsv, *hsv_min, *hsv_max)
    else:
        mask = cv2.inRange(roi_hsv, np.array(hsv_min), np.array(hsv_max))
    
    # 应用形态学操作减少噪点
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找出最大轮廓
    max_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > contour_area_threshold:
            max_area = area
            max_contour = contour
    
    # 显示掩码（如果启用）
    if show_mask:
        if roi and len(roi) == 4:
            x1, y1, x2, y2 = roi
            output_frame[y1:y2, x1:x2] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            output_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # 如果找到有效轮廓，绘制并标记
    if max_contour is not None:
        # 计算轮廓的中心
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 如果使用ROI，调整坐标
            if roi and len(roi) == 4:
                cx += roi[0]
                cy += roi[1]
                max_contour = max_contour + np.array([[roi[0], roi[1]]])
            
            # 绘制轮廓和中心点
            cv2.drawContours(output_frame, [max_contour], -1, COLORS[selected_color]['rgb'], 2)
            cv2.circle(output_frame, (cx, cy), 5, COLORS[selected_color]['rgb'], -1)
            
            # 添加颜色标签
            cv2.putText(output_frame, f"{selected_color.upper()}", (cx, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[selected_color]['rgb'], 2)
            
            # 存储检测结果用于平滑处理
            detection_history.append(selected_color)
            if len(detection_history) > history_size:
                detection_history.pop(0)
            
            # 计算检测历史中出现最多的颜色
            if detection_enabled:
                from collections import Counter
                if len(detection_history) > 0:
                    most_common_color = Counter(detection_history).most_common(1)[0][0]
                    cv2.putText(output_frame, f"Detected: {most_common_color.upper()}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[most_common_color]['rgb'], 2)
    
    # 显示当前选择的颜色
    cv2.putText(output_frame, f"Selected: {selected_color.upper()}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[selected_color]['rgb'], 2)
    
    # 显示控制信息
    cv2.putText(output_frame, "Press 'd' to toggle detection, 'm' to show mask, 'r' to reset ROI", 
                (10, output_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return output_frame

def main():
    global detection_enabled, show_mask, roi, selected_color
    
    # 检查Python版本
    if sys.version_info.major == 2:
        print('Please run this program with python3!')
        sys.exit(0)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        sys.exit(0)
    
    # 创建显示窗口
    cv2.namedWindow('Color Detection')  # 确保窗口存在后再设置回调
    
    # 设置鼠标回调
    cv2.setMouseCallback('Color Detection', select_roi)
    
    # 创建滑动条
    create_trackbars()
    
    print("颜色识别程序已启动")
    print("按 'd' 键切换检测模式")
    print("按 'm' 键显示/隐藏掩码")
    print("按 'r' 键重置ROI")
    print("按 'q' 键退出程序")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头图像")
            break
        
        # 处理帧
        processed_frame = process_frame(frame)
        
        # 显示结果
        cv2.imshow('Color Detection', processed_frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 退出程序
            break
        elif key == ord('d'):  # 切换检测模式
            detection_enabled = not detection_enabled
            print(f"检测模式: {'开启' if detection_enabled else '关闭'}")
        elif key == ord('m'):  # 显示/隐藏掩码
            show_mask = not show_mask
            print(f"掩码显示: {'开启' if show_mask else '关闭'}")
        elif key == ord('r'):  # 重置ROI
            roi = None
            print("ROI已重置")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
