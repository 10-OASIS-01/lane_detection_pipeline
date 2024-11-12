"""
lane_detection_pipeline/binary_thresholds.py

This module includes various thresholding functions for creating binary images
focused on lane line features. Functions here convert color spaces and apply
gradients to highlight lane lines for further processing.
"""
import cv2
import numpy as np


def hls_l_select(img, thresh=(220, 255)):
    """
    将输入图像转换为 HLS 色彩空间，并基于亮度通道生成二值化图像。

    参数：
    - img: 输入图像，格式为 BGR
    - thresh: 亮度通道的阈值范围，默认为 (220, 255)

    返回：
    - binary_output: 二值化后的图像
    """
    # 检查输入图像的有效性
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须为图像的 numpy 数组。")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须为三通道 BGR 图像。")

    # 转换图像至 HLS 色彩空间并提取亮度通道
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]

    # 缩放亮度通道以保持在 0 到 255 之间
    if np.max(l_channel) > 0:
        l_channel = (l_channel * (255.0 / np.max(l_channel))).astype(np.uint8)

    # 应用阈值生成二值化图像
    binary_output = np.zeros_like(l_channel, dtype=np.uint8)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 255

    return binary_output


# 亮度划分函数
def hls_s_select(img, thresh=(125, 255)):
    """
    将输入图像转换为 HLS 色彩空间，并基于饱和度通道生成二值化图像。

    参数：
    - img: 输入图像，格式为 BGR
    - thresh: 饱和度通道的阈值范围，默认为 (125, 255)

    返回：
    - binary_output: 二值化后的图像
    """
    # 输入检查：确保图像为 BGR 格式的 numpy 数组
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须为图像的 numpy 数组。")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须为三通道 BGR 图像。")

    # 转换图像至 HLS 色彩空间并提取饱和度通道
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    # 缩放饱和度通道至 0 到 255 范围内
    max_val = np.max(s_channel)
    if max_val > 0:
        s_channel = (s_channel * (255.0 / max_val)).astype(np.uint8)

    # 应用阈值生成二值化图像
    binary_output = np.zeros_like(s_channel, dtype=np.uint8)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    应用 Sobel 算子的梯度方向阈值以生成二值化图像。

    参数：
    - img: 输入图像，格式为 BGR
    - sobel_kernel: Sobel 算子的核大小，默认为 3
    - thresh: 梯度方向阈值范围，默认为 (0, π/2)

    返回：
    - binary_output: 满足方向阈值的二值化图像
    """
    # 输入检查：确保图像为 BGR 格式的 numpy 数组
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须为图像的 numpy 数组。")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须为三通道 BGR 图像。")

    # 转换图像至灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算 x 和 y 方向的 Sobel 梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 计算 x 和 y 梯度的绝对值
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 计算梯度方向
    direction_sobelxy = np.arctan2(abs_sobely, abs_sobelx)

    # 创建满足方向阈值的二值化图像
    binary_output = np.zeros_like(direction_sobelxy, dtype=np.uint8)
    binary_output[(direction_sobelxy >= thresh[0]) & (direction_sobelxy <= thresh[1])] = 1

    return binary_output


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    应用梯度幅度阈值生成二值化图像。

    参数：
    - img: 输入图像，格式为 BGR
    - sobel_kernel: Sobel 算子的核大小，默认为 3
    - mag_thresh: 幅度阈值范围，默认为 (0, 255)

    返回：
    - binary_output: 满足幅度阈值的二值化图像
    """
    # 输入检查：确保图像为 BGR 格式的 numpy 数组
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须为图像的 numpy 数组。")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须为三通道 BGR 图像。")

    # 转换图像至灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算 x 和 y 方向的 Sobel 梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 计算梯度幅度
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 缩放梯度幅度至 0-255 范围并转换为 uint8 类型
    max_val = np.max(abs_sobelxy)
    if max_val > 0:
        scaled_sobelxy = np.uint8(255 * abs_sobelxy / max_val)
    else:
        scaled_sobelxy = np.zeros_like(abs_sobelxy, dtype=np.uint8)

    # 应用幅度阈值生成二值化图像
    binary_output = np.zeros_like(scaled_sobelxy, dtype=np.uint8)
    binary_output[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1

    return binary_output


def abs_sobel_threshold(img, orient='x', thresh_min=0, thresh_max=255):
    """
    对输入图像应用 Sobel 算子，并基于梯度绝对值进行阈值处理。

    参数：
    - img: 输入图像，格式为 BGR
    - orient: 梯度方向 ('x' 或 'y')
    - thresh_min: 阈值下限
    - thresh_max: 阈值上限

    返回：
    - binary_output: 应用阈值后的二值化图像
    """
    # 输入检查：确保图像为 BGR 格式的 numpy 数组
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须为图像的 numpy 数组。")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须为三通道 BGR 图像。")
    if orient not in ['x', 'y']:
        raise ValueError("参数 orient 必须为 'x' 或 'y'。")

    # 转换图像至灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 根据指定方向计算 Sobel 梯度的绝对值
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    else:  # orient == 'y'
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # 将梯度缩放到 8 位范围 (0-255)
    max_val = np.max(abs_sobel)
    if max_val > 0:
        scaled_sobel = np.uint8(255 * abs_sobel / max_val)
    else:
        scaled_sobel = np.zeros_like(abs_sobel, dtype=np.uint8)

    # 应用阈值生成二值化图像
    binary_output = np.zeros_like(scaled_sobel, dtype=np.uint8)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


# Lab蓝黄通道划分函数
def lab_b_select(img, thresh=(215, 255)):
    """
    转换图像至 LAB 色彩空间并基于蓝黄通道 (b 通道) 生成二值化图像。

    参数：
    - img: 输入图像，格式为 BGR
    - thresh: b 通道的阈值范围，默认为 (215, 255)

    返回：
    - binary_output: 满足阈值的二值化图像
    """
    # 输入检查：确保图像为 BGR 格式的 numpy 数组
    if not isinstance(img, np.ndarray):
        raise TypeError("输入必须为图像的 numpy 数组。")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("输入图像必须为三通道 BGR 图像。")

    # 转换图像至 LAB 色彩空间并提取 b 通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:, :, 2]

    # 归一化 b 通道以保持在 0 到 255 之间（若图像中存在显著黄色）
    max_val = np.max(lab_b)
    if max_val > 100:
        lab_b = (lab_b * (255.0 / max_val)).astype(np.uint8)

    # 应用阈值生成二值化图像
    binary_output = np.zeros_like(lab_b, dtype=np.uint8)
    binary_output[(lab_b > thresh[0]) & (lab_b <= thresh[1])] = 255

    return binary_output
