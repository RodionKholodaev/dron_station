import numpy as np
import cv2
import sys
import requests
import threading
import time

from esp32_camera_stream import ESP32CameraStream
from markers_conf import MARKER_CENTERS, MARKER_SIZE
from camera_conf import ESP32_URL

def get_global_camera_pose(corners, ids, mtx, dist):
    obj_points_all = []
    img_points_all = []
    
    s = MARKER_SIZE / 2

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in MARKER_CENTERS:
            continue  # Пропускаем маркеры, которых нет в нашей карте
            
        # Берем центр конкретного маркера из словаря
        c = MARKER_CENTERS[marker_id]
        
        # Вычисляем мировые координаты 4-х углов этого маркера
        # Важно: порядок углов должен строго соответствовать выходу ArUco (CW: TL, TR, BR, BL)
        marker_obj_pts = np.array([
            [c[0] - s, c[1] + s, c[2]], # Top-Left
            [c[0] + s, c[1] + s, c[2]], # Top-Right
            [c[0] + s, c[1] - s, c[2]], # Bottom-Right
            [c[0] - s, c[1] - s, c[2]]  # Bottom-Left
        ], dtype=np.float32)
        
        obj_points_all.append(marker_obj_pts)
        img_points_all.append(corners[i].reshape(4, 2))

    if len(obj_points_all) == 0:
        return None, None, None

    # Объединяем все точки в общие массивы
    obj_points_all = np.vstack(obj_points_all)
    img_points_all = np.vstack(img_points_all)

    # Решаем PnP для всей группы точек сразу
    # Если точек много (больше 1 маркера), используем стандартный флаг или SQPNP
    success, rvec, tvec = cv2.solvePnP(obj_points_all, img_points_all, mtx, dist)

    if not success:
        return None, None, None

    # Инвертируем трансформацию для получения позиции камеры в мире
    rmat, _ = cv2.Rodrigues(rvec)
    rmat_t = rmat.T
    camera_pos = -rmat_t @ tvec.reshape(3, 1)
    
    return camera_pos.flatten(), rvec, tvec

def pose_estimation(frame, aruco_dict, detector_params, k, d):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Получаем одну общую позицию относительно "мира"
        cam_pos, rvec, tvec = get_global_camera_pose(corners, ids, k, d)
        
        if cam_pos is not None:
            x, y, z = cam_pos
            text = f"Drone Pos: X={x:.2f} Y={y:.2f} Z={z:.2f}"
            print(text)
            
            # Рисуем оси в центре координат мира (в точке 0,0,0)
            cv2.drawFrameAxes(frame, k, d, rvec, tvec, 0.2)

    return frame


# --- 3. Основной блок запуска ---
if __name__ == '__main__':

    # Загрузка калибровки (убедитесь, что пути верны)
    try:
        k = np.load("camera_params/calibration_matrix.npy")
        d = np.load("camera_params/distortion_coefficients.npy")
    except FileNotFoundError:
        print("Ошибка: Файлы калибровки не найдены!")
        sys.exit(0)

    # Инициализация ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()

    # Запуск камеры
    print(f"Подключение к ESP32: {ESP32_URL}...")
    cam = ESP32CameraStream(ESP32_URL).start()

    while True:
        frame = cam.read()

        if frame is None:
            continue

        # Обработка кадра
        output_frame = pose_estimation(frame, aruco_dict, detector_params, k, d)

        # Показ результата
        cv2.imshow('ESP32 Pose Estimation', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()