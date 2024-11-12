"""
lane_detection_pipeline/lane_detection.py

This module implements functions for detecting lane line pixels using sliding windows
and polynomial fitting. It includes functions to locate lane lines, fit polynomial curves,
and track lanes across frames for a continuous lane detection.
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    在二值化的鸟瞰图中找到左右车道线像素。

    参数：
    - binary_warped: 二值化的鸟瞰图图像
    - nwindows: 滑动窗口的数量
    - margin: 窗口的宽度的一半
    - minpix: 用于重新定位窗口的最小像素数

    返回：
    - leftx, lefty: 左车道线像素的 x 和 y 坐标
    - rightx, righty: 右车道线像素的 x 和 y 坐标
    - out_img: 带有滑动窗口可视化的图像
    """
    # 输入检查
    if not isinstance(binary_warped, np.ndarray) or binary_warped.ndim != 2:
        raise ValueError("输入必须是二值化的单通道图像。")

    # 计算底部一半区域的直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 窗口的高度
    window_height = binary_warped.shape[0] // nwindows

    # 获取所有非零像素的坐标
    nonzeroy, nonzerox = binary_warped.nonzero()

    # 当前窗口的左右车道位置
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 初始化左右车道线像素索引的列表
    left_lane_inds = []
    right_lane_inds = []

    # 创建输出图像用于可视化
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # 遍历每一个窗口
    for window in range(nwindows):
        # 定义窗口边界
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 绘制窗口边界在可视化图像上
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # 获取窗口内的像素索引
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # 添加找到的像素索引
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果找到的像素数量超过最小值，则重新确定窗口位置
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # 将索引列表拼接成数组
    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    # 提取左、右车道线像素位置
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    在二值化的鸟瞰图中找到车道线像素并拟合二次多项式以生成车道线曲线。

    参数：
    - binary_warped: 二值化的鸟瞰图图像
    - nwindows: 滑动窗口的数量
    - margin: 窗口的宽度的一半
    - minpix: 用于重新定位窗口的最小像素数

    返回：
    - out_img: 可视化结果图像
    - left_fit: 左车道线的二次多项式系数
    - right_fit: 右车道线的二次多项式系数
    - ploty: 用于绘图的 y 值
    """
    # 先找到车道线像素
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, nwindows, margin, minpix)

    # 使用 np.polyfit 拟合二次多项式
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成用于绘图的 y 值
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # 计算拟合曲线上的 x 值
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # 可视化：在车道线位置上着色
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty


# Step 6 : Track lane lines based latest lane line result
#################################################################
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    """
    使用二次多项式拟合左、右车道线，并生成拟合曲线的 x 和 y 值用于绘图。

    参数：
    - img_shape: 图像的形状，用于确定 y 值的范围
    - leftx, lefty: 左车道线像素的 x 和 y 坐标
    - rightx, righty: 右车道线像素的 x 和 y 坐标

    返回：
    - left_fitx: 左车道线拟合曲线的 x 值
    - right_fitx: 右车道线拟合曲线的 x 值
    - ploty: 用于绘图的 y 值
    - left_fit: 左车道线的二次多项式系数
    - right_fit: 右车道线的二次多项式系数
    """
    # 确保输入坐标有效
    if not (len(leftx) and len(lefty) and len(rightx) and len(righty)):
        raise ValueError("输入坐标不能为空")

    # 使用 np.polyfit 拟合二次多项式
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成用于绘图的 y 值
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    # 计算拟合曲线上的 x 值
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit, margin=60):
    """
    在之前拟合的多项式曲线附近查找车道线像素，并重新拟合车道线。

    参数：
    - binary_warped: 二值化的鸟瞰图图像
    - left_fit: 左车道线的二次多项式系数
    - right_fit: 右车道线的二次多项式系数
    - margin: 搜索窗口宽度的一半，默认为 60

    返回：
    - result: 带有车道线和搜索区域的可视化图像
    - left_fit: 左车道线的更新多项式系数
    - right_fit: 右车道线的更新多项式系数
    - ploty: 用于绘图的 y 值
    """
    # 获取所有非零像素坐标
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    # 定义左右车道线的搜索区域
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # 提取左右车道线的像素坐标
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # 拟合新的多项式
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # 可视化
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # 将左右车道线像素上色
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # 生成搜索窗口区域
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # 将搜索区域填充为绿色
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, left_fit, right_fit, ploty

