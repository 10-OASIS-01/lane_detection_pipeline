"""
lane_detection_pipeline/drawing_utils.py

This module contains utility functions for drawing lane lines on images.
It includes functions to fill lane areas, overlay lane markings on the
original image, and add curvature and offset information as text annotations.
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def drawing(undist, bin_warped, left_fitx, right_fitx, ploty, Minv):
    """
    在透视变换后的图像上绘制车道线填充区域，并将其逆透视投影回原图。

    参数：
    - undist: 去畸变的原始图像
    - bin_warped: 二值化的透视变换图像
    - left_fitx: 左车道线拟合曲线的 x 坐标
    - right_fitx: 右车道线拟合曲线的 x 坐标
    - ploty: 用于绘图的 y 坐标数组
    - Minv: 透视变换的逆矩阵

    返回：
    - result: 带有绘制车道线填充区域的图像
    """
    # 创建用于绘制车道线填充区域的空白图像
    warp_zero = np.zeros_like(bin_warped, dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # 准备左、右车道线顶点并连接为一个多边形区域
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # 在透视变换的空白图像上填充车道线区域
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # 将填充区域逆透视变换回原图空间
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # 将填充区域与原始图像叠加
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def draw_text(image, curverad, offset):
    """
    在图像上绘制曲率半径和偏移量信息。

    参数：
    - image: 要绘制文本的图像
    - curverad: 车道曲率半径（米）
    - offset: 车辆相对车道中心的偏移量（米）

    返回：
    - result: 带有曲率半径和偏移量文本的图像
    """
    result = np.copy(image)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 绘制曲率半径文本
    curvature_text = f'Curve Radius: {curverad:.2f} m'
    cv2.putText(result, curvature_text, (20, 50), font, 1.2, (255, 255, 255), 2)

    # 绘制偏移量文本
    if offset > 0:
        offset_text = f'Right of center: {abs(offset):.3f} m'
    else:
        offset_text = f'Left of center: {abs(offset):.3f} m'
    cv2.putText(result, offset_text, (20, 100), font, 1.2, (255, 255, 255), 2)

    return result

