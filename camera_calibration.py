"""
lane_detection_pipeline/camera_calibration.py

This module contains functions for camera calibration and image undistortion.
It includes functions to read chessboard patterns, compute calibration coefficients,
and remove distortion from input images based on calculated parameters.
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def get_camera_calibration_coefficients(chessboard_pattern, nx, ny):
    """
    读入图像、进行相机校准并返回校准系数。

    参数：
    - chessboard_pattern: 棋盘格图片文件的通配符路径，例如 'images/*.jpg'
    - nx: 棋盘格的内角点列数
    - ny: 棋盘格的内角点行数

    返回：
    - ret: 校准结果标志
    - mtx: 相机内参矩阵
    - dist: 畸变系数
    - rvecs: 旋转向量
    - tvecs: 平移向量
    """
    # 准备3D棋盘格点，如 (0,0,0), (1,0,0), ... (nx-1, ny-1, 0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # 存储3D对象点和2D图像点
    objpoints = []
    imgpoints = []

    # 读取所有棋盘格图像
    images = glob.glob(chessboard_pattern)
    if not images:
        print("未找到任何图像文件，请检查文件路径或文件格式。")
        return None

    print(f"找到 {len(images)} 张图像用于校准。")

    img_size = None  # 图像尺寸

    # 检测每张图像的棋盘格角点
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"无法加载图像 {fname}，跳过该文件。")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = (img.shape[1], img.shape[0])

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"未检测到图像 {fname} 的棋盘格角点，跳过该文件。")

    # 进行相机校准
    if objpoints and imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        print("校准成功！")
        return ret, mtx, dist, rvecs, tvecs
    else:
        print("未检测到足够的棋盘格角点进行校准，请检查输入图像。")
        return None


def undistort_image(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)

