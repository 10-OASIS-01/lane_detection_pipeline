"""
lane_detection_pipeline/curvature_offset.py

This module provides functions for calculating lane curvature in meters
and the vehicle's offset from the lane center. These measurements are crucial
for evaluating the car's position and road curvature.
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def measure_curvature_real(left_fit_cr, right_fit_cr, ploty, ym_per_pix=30/720, xm_per_pix=3.7/700):
    """
    计算以米为单位的车道曲率半径和车辆偏移量。

    参数：
    - left_fit_cr: 左车道线的二次多项式系数
    - right_fit_cr: 右车道线的二次多项式系数
    - ploty: y 值的数组，用于评估曲率
    - ym_per_pix: y 方向上像素与实际距离的转换比例，默认为 30/720 (米/像素)
    - xm_per_pix: x 方向上像素与实际距离的转换比例，默认为 3.7/700 (米/像素)

    返回：
    - left_curverad: 左车道线的曲率半径（米）
    - right_curverad: 右车道线的曲率半径（米）
    - offset: 车辆相对车道中心的偏移量（米）
    """
    # 使用图像底部的最大 y 值来计算曲率半径
    y_eval = np.max(ploty)

    # 计算左右车道线的曲率半径
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * right_fit_cr[0])

    # 计算车道中心与图像中心的偏移量
    left_position = left_fit_cr[0] * (y_eval ** 2) + left_fit_cr[1] * y_eval + left_fit_cr[2]
    right_position = right_fit_cr[0] * (y_eval ** 2) + right_fit_cr[1] * y_eval + right_fit_cr[2]
    lane_center = (left_position + right_position) / 2
    image_center = 1280 / 2  # 假设图像宽度为 1280 像素
    offset = (image_center - lane_center) * xm_per_pix

    return left_curverad, right_curverad, offset

