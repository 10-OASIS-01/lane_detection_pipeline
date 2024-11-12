"""
lane_detection_pipeline/perspective_transform.py

This module provides functionality for performing perspective transformations on images.
It includes functions to warp and inverse warp images based on specified source
and destination points, enabling a bird's-eye view for lane detection.
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def warp_image(image, src_points, dst_points):
    """
    使用源点和目标点进行透视变换以获得变换后的图像。

    参数：
    - image: 输入图像
    - src_points: 源点坐标，格式为 np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    - dst_points: 目标点坐标，格式与 src_points 相同

    返回：
    - warped_image: 透视变换后的图像
    - M: 透视变换矩阵
    - Minv: 逆透视变换矩阵
    """
    # 检查输入点的类型和形状
    if not isinstance(src_points, np.ndarray) or not isinstance(dst_points, np.ndarray):
        raise TypeError("src_points 和 dst_points 必须为 numpy 数组类型。")
    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("src_points 和 dst_points 必须具有形状 (4, 2) 的四个坐标点。")

    # 图像大小
    image_size = (image.shape[1], image.shape[0])

    # 计算透视变换矩阵和逆矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    # 应用透视变换
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

    return warped_image, M, Minv

